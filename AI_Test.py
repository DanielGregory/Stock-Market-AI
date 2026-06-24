import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import time as tm
import os
import pickle
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

Finnhub_API_Key = os.getenv("FINNHUB_API_KEY", "")
PORTFOLIO_SIZE = float(os.getenv("PORTFOLIO_SIZE", "100000"))

def fetch_historical_data(symbol, days=200):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1d")
        df.reset_index(inplace=True)
        df.rename(columns={'Date': 'Timestamp'}, inplace=True)
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.tz_localize(None)
        return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def add_features(data):
    try:
        data['Previous_Close'] = data['Close'].shift(1)

        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility_15'] = data['Close'].rolling(window=15).std()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']
        data['Volume_5'] = data['Volume'].rolling(window=5).mean()
        data['Percent_Change'] = data['Close'].pct_change()
        data['Yesterday_Percent_Change'] = data['Percent_Change'].shift(1)

        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        obv = [0]
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        data['Obv'] = obv

        data['12_EMAs'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['26_EMAs'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['12_EMAs'] - data['26_EMAs']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Crossover'] = (data['MACD'] > data['Signal_Line']).astype(int)
        data['MACD_5'] = data['MACD_Crossover'].rolling(window=5).mean()

        data['Next_Close'] = data['Close'].shift(-1)
        data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)
        data['Direction_5'] = data['Direction'].shift(1).rolling(window=5).mean()

        data['TR'] = data.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Previous_Close']),
                abs(row['Low'] - row['Previous_Close'])
            ),
            axis=1
        )
        data['ATR_14'] = data['TR'].rolling(window=14).mean()

        data.dropna(inplace=True)
        return data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()

def train_and_predict(data, symbol, test_size=50):
    try:
        train_data = data.iloc[:-test_size, :]
        test_data = data.iloc[-test_size:, :]

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                    'Volatility_20', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'Volume_5']

        X_train = train_data[features]
        y_train = train_data['Direction'].values
        X_test = test_data[features]
        y_test = test_data['Direction'].values

        # Capture actual close prices before scaling so profit calc uses real prices
        close_prices_test = test_data['Close'].values

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        stock_model_dir = os.path.join(model_dir, symbol)
        os.makedirs(stock_model_dir, exist_ok=True)
        model_path = os.path.join(stock_model_dir, f"{symbol}_sgd_model.pkl")

        best_cv_f1_score = 0  # default; only updated when training a new model

        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded existing model for {symbol}.")
        else:
            model = SGDClassifier(random_state=42)
            param_grid = {
                'loss': ['hinge', 'log_loss'],
                'penalty': ['l1', 'l2'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['invscaling', 'adaptive'],
                'eta0': [0.01, 0.1, 0.001]
            }
            classes = np.unique(y_train)
            print(f"Initialized new model for {symbol}.")
            tscv = TimeSeriesSplit(n_splits=5)
            grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='f1', verbose=1, n_jobs=1)
            grid_search.fit(X_train, y_train)
            best_cv_f1_score = grid_search.best_score_
            model = grid_search.best_estimator_
            print(f"Best CV F1 Score: {best_cv_f1_score:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")

        classes = np.unique(y_train)
        batch_size = 20
        num_samples = X_train.shape[0]
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            if i == 0 and not os.path.exists(model_path):
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved updated model for {symbol}.")

        feature_names = features
        weights = model.coef_[0]
        feature_weights = pd.DataFrame({'Feature': feature_names, 'Weight': weights})
        feature_weights['Abs_Weight'] = feature_weights['Weight'].abs()
        feature_weights = feature_weights.sort_values(by='Abs_Weight', ascending=False)
        print(feature_weights[['Feature', 'Weight']])

        top_feature = feature_weights.iloc[0]
        top_feature_name = top_feature['Feature']
        top_feature_weight = top_feature['Weight']

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy for {symbol}: {accuracy:.2f}")

        last_10_predictions = predictions[-10:]
        last_10_actuals = y_test[-10:]
        last_10_accuracy = accuracy_score(last_10_actuals, last_10_predictions)

        return {
            "model": model,
            "y_test": y_test,
            "predictions": predictions,
            "accuracy": accuracy,
            "close_prices_test": close_prices_test,
            "top_feature": top_feature_name,
            "top_weight": top_feature_weight,
            "last_10_accuracy": last_10_accuracy,
            "best_cv_f1_score": best_cv_f1_score,
            "last_10_predictions": last_10_predictions
        }

    except Exception as e:
        print(f"Error during training and prediction for {symbol}: {e}")
        return None

def get_current_price(data):
    try:
        if not data.empty:
            last_close = data['Close'].iloc[-1]
            last_volume = data['Volume'].iloc[-1]
            last_volume_5 = data['Volume_5'].iloc[-1]
            return last_close, last_volume, last_volume_5
        else:
            print("Data is empty, cannot fetch the last close price.")
            return None, None, None
    except KeyError:
        print("The DataFrame does not contain a 'Close' column.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while fetching the last close price: {e}")
        return None, None, None

def calculate_quantity(symbol, allocation_percent, principal, current_price):
    try:
        allocation = principal * (allocation_percent / 100)
        if current_price is None:
            raise ValueError(f"Could not fetch price for {symbol}")
        quantity = int(allocation // current_price)
        print(f"Principal: ${principal:.2f}, Allocation: ${allocation:.2f}, Quantity: {quantity}")
        return quantity
    except Exception as e:
        print(f"Error calculating quantity: {e}")
        return 0

def calculate_atr(data, period=14):
    if len(data) < period:
        return None
    data['TR'] = data.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['Previous_Close']),
            abs(row['Low'] - row['Previous_Close'])
        ),
        axis=1
    )
    atr = data['TR'].rolling(window=period).mean().iloc[-1]
    return atr

def calculate_profit(predictions, y_test, close_prices, quantity):
    try:
        profit = 0.0
        for i in range(len(y_test) - 1):
            current_close = close_prices[i]
            next_close = close_prices[i + 1]
            predicted_direction = predictions[i]
            if predicted_direction == 1:
                if next_close > current_close:
                    profit += (next_close - current_close) * quantity
                else:
                    profit -= (current_close - next_close) * quantity
            elif predicted_direction == 0:
                if next_close < current_close:
                    profit += (current_close - next_close) * quantity
                else:
                    profit -= (next_close - current_close) * quantity
        print(f"Total profit/loss from all predictions: ${profit:.2f}")
        return profit
    except Exception as e:
        print(f"Error calculating profit: {e}")

def fetch_finnhub_news(symbol, finnhub_api_key, days=200):
    today = datetime.today()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching news data:", response.json())
        return None
    news_data = response.json()
    news_df = pd.DataFrame(news_data)
    if news_df.empty:
        print("No news articles found.")
        return None
    news_df['Timestamp'] = pd.to_datetime(news_df['datetime'], unit='s')
    news_df['Timestamp'] = news_df['Timestamp'].dt.floor('D')
    news_df['Timestamp'] = news_df['Timestamp'].dt.tz_localize('UTC')
    news_df['Timestamp'] = news_df['Timestamp'].dt.tz_convert('America/New_York').dt.normalize()
    news_df['Timestamp'] = news_df['Timestamp'].dt.tz_localize(None)
    return news_df[['Timestamp', 'headline', 'summary']]

def analyze_sentiment(news_df):
    if news_df is None:
        return None
    analyzer = SentimentIntensityAnalyzer()
    news_df['sentiment'] = news_df['headline'].apply(
        lambda text: analyzer.polarity_scores(text)['compound']
    )
    return news_df

def aggregate_sentiment(data, news_df):
    if news_df is None:
        data['sentiment'] = 0
        last_row = data.iloc[[-1]]
        return data, last_row
    sentiment_by_date = news_df.groupby('Timestamp')['sentiment'].mean().reset_index()
    merged = pd.merge(data, sentiment_by_date, how='left', on='Timestamp')
    merged['sentiment'] = merged['sentiment'].fillna(0)
    tm.sleep(1)
    last_row = merged.iloc[[-1]]
    return merged, last_row


try:
    sp500_symbols = list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'])
except Exception:
    sp500_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
    print("Could not fetch S&P 500 list, using default symbols.")

stocks = sp500_symbols
allocation_percent = 5
model_dir = "Models_SGD"
os.makedirs(model_dir, exist_ok=True)


def main_trading_logic():
    results = []
    principal = PORTFOLIO_SIZE
    flipped = False

    for symbol in stocks:
        try:
            print(f"{datetime.now()}: Fetching data for {symbol}...")
            bars_df = fetch_historical_data(symbol)
            data = add_features(bars_df)

            if Finnhub_API_Key:
                news_df = fetch_finnhub_news(symbol, Finnhub_API_Key)
                news_df = analyze_sentiment(news_df)
                merged, last_row = aggregate_sentiment(data, news_df)
            else:
                data['sentiment'] = 0
                merged = data
                last_row = data.iloc[[-1]]

            result = train_and_predict(merged, symbol)

            if result:
                predictions = result.get("predictions")
                accuracy = result.get("accuracy")
                close_prices_test = result.get("close_prices_test")
                y_test = result.get("y_test")
                top_feature = result.get("top_feature")
                top_weight = result.get("top_weight")
                Last_10_accuracy = result.get("last_10_accuracy")
                best_cv_f1_score = result.get("best_cv_f1_score")
                pred_last_10 = result.get("last_10_predictions")

                if accuracy <= 0.48:
                    accuracy = 1 - accuracy
                    predictions = 1 - predictions
                    Last_10_accuracy = 1 - Last_10_accuracy
                    pred_last_10 = 1 - pred_last_10
                    flipped = True
                    print(f"Flipped predictions for {symbol} (accuracy now {accuracy:.2f})")
                else:
                    flipped = False
            else:
                print(f"Error during training and prediction for {symbol}")
                continue

            atr_14 = calculate_atr(data, period=14)
            current_price, volume, volume_5 = get_current_price(last_row)
            quantity = calculate_quantity(symbol, allocation_percent, principal, current_price)

            profit = calculate_profit(predictions, y_test, close_prices_test, quantity)
            profit10 = calculate_profit(pred_last_10, y_test[-10:], close_prices_test[-10:], quantity)

            results.append({
                "Symbol": symbol,
                "Accuracy": accuracy * 100,
                "Accuracy_10": Last_10_accuracy * 100,
                "Profit": profit,
                "Profit10": profit10,
                "Flipped": flipped,
                "CV_Score": best_cv_f1_score * 100,
                "Volume": volume,
                "Volume_5": volume_5,
                "top_feature": top_feature,
                "top_weight": top_weight,
                "ATR_14": atr_14,
                "Current_Price": round(current_price, 2) if current_price else "N/A"
            })

            try:
                output_file = "SGD_results.xlsx"
                if os.path.exists(output_file):
                    existing_df = pd.read_excel(output_file)
                    out_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
                    out_df = out_df.drop_duplicates(subset=["Symbol"], keep="last")
                else:
                    out_df = pd.DataFrame(results)
                out_df.to_excel(output_file, index=False)
                print(f"Results saved to {output_file}")
            except Exception as e:
                print(f"Error saving results to Excel: {e}")

            if accuracy >= 0.54:
                print(f"{datetime.now()}: Accuracy sufficient ({accuracy:.2f}) for {symbol}.")
            else:
                print(f"{datetime.now()}: Accuracy insufficient ({accuracy:.2f}) for {symbol}, skipping.")

        except Exception as e:
            print(f"Error in loop for {symbol}: {e}")

if __name__ == "__main__":
    while True:
        try:
            main_trading_logic()
            tm.sleep(10000)
        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)
