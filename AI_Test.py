#import the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import alpaca_trade_api as tradeapi
import time as tm
import os
import pickle
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler



#Alpaca API key
os.environ["APCA_API_KEY_ID"] = "Your API Key ID"
os.environ["APCA_API_SECRET_KEY"] = "Your API Secret Key"
BASE_URL = "https://paper-api.alpaca.markets"

def fetch_historical_data(api, symbol):
    """
    Fetch historical stock data for the given symbol.

    Parameters:
        api: Alpaca API object.
        symbol (str): The stock symbol to fetch data for.
        timeframe_minutes (int): The granularity of the data in minutes.

    Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
    """
    try:
        # Set the start and end dates for the data fetch
        end_date = (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ')
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Fetch bars using the specified timeframe
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date, end=end_date, limit=1000)

        # Convert list of Bar objects to a DataFrame
        bar_data = []
        for bar in bars:
            bar_data.append({
                'Timestamp': bar.t,
                'Open': bar.o,
                'Volume': bar.v,
                'High': bar.h,
                'Low': bar.l,
                'Close': bar.c,
            })

        # Create a pandas DataFrame
        bars_df = pd.DataFrame(bar_data)
        return bars_df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def add_features(data):
    """
    Add features to the stock data for analysis or modeling.

    Parameters:
        data (pd.DataFrame): A DataFrame containing stock price data with a 'Close' column.

    Returns:
        pd.DataFrame: A DataFrame with added features and cleaned of missing values.
    """
    try:
       # Create the Previous_Close feature
        data['Previous_Close'] = data['Close'].shift(1)
       
        # Add additional features
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility_15'] = data['Close'].rolling(window=15).std()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']
        data['Volume_5'] = data['Volume'].rolling(window=5).mean()
        data['Percent_Change'] = data['Close'].pct_change()
        data['Yesterday_Percent_Change']= data['Percent_Change'].shift(1)  # Fill NaN values with 0

        # Calculate RSI (Relative Strength Index)
        delta = data['Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        obv = [0]  # Start OBV at 0
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i - 1]:  # Price up
                obv.append(obv[-1] + data['Volume'].iloc[i])
            elif data['Close'].iloc[i] < data['Close'].iloc[i - 1]:  # Price down
                obv.append(obv[-1] - data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])  # Price unchanged

        # Add OBV as a feature
        data['Obv'] = obv

        data['12_EMAs'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['26_EMAs'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = data['12_EMAs'] - data['26_EMAs']
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Crossover'] = (data['MACD'] > data['Signal_Line']).astype(int)
        data['MACD_5'] = data['MACD_Crossover'].rolling(window=5).mean()

        # Create the Direction target
        data['Next_Close'] = data['Close'].shift(-1)
        data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)  # 1 if increase, 0 if decrease
        data['Direction_5'] = data['Direction'].shift(1).rolling(window=5).mean()

         # Calculate ATR (Average True Range)
        data['TR'] = data.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Previous_Close']),
                abs(row['Low'] - row['Previous_Close'])
            ),
            axis=1
        )
        data['ATR_14'] = data['TR'].rolling(window=14).mean()


        # Drop any rows with missing values
        data.dropna(inplace=True)

        return data
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()
   
def train_and_predict(data, symbol, test_size=50):
    """
    Train an XGBClassifier model and make predictions on the test data.

    Parameters:
        data (pd.DataFrame): The dataset containing features and target.
        features (list): List of feature column names.
        target (str): Name of the target column.
        test_size (int): Number of samples to reserve for testing.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    try:
        # Split the data into training and testing sets
        train_data = data.iloc[:-test_size, :]
        test_data = data.iloc[-test_size:, :]

        X_train = train_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                               'Volatility_20', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'Volume_5']]
        y_train = train_data['Direction']
        X_test = test_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                               'Volatility_20', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'Volume_5']]
        y_test = test_data['Direction']

        # Scale the data (SGDClassifier benefits from normalized data)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        scaler_y = MinMaxScaler()
        y_train = scaler_y.fit_transform(y_train)
        y_test = scaler_y.transform(y_test)



        # Create a subdirectory for each stock's model
        stock_model_dir = os.path.join(model_dir, symbol)
        os.makedirs(stock_model_dir, exist_ok=True)
        model_path = os.path.join(stock_model_dir, f"{symbol}_sgd_model.pkl")

        # Load or initialize the model
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded existing model for {symbol}.")
        else:
            # Initialize a new SGDClassifier (set the appropriate hyperparameters)
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
            print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
            print(f"Best Parameters: {grid_search.best_params_}")

        
        # Incrementally train the model
        batch_size = 20
        num_samples = X_train.shape[0]

        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            # Ensure the first batch includes `classes=np.unique(y_train)`
            if i == 0 and not os.path.exists(model_path):
                model.partial_fit(X_batch, y_batch, classes=classes)
            else:
                model.partial_fit(X_batch, y_batch)
        # Save the updated model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved updated model for {symbol}.")


        # List of feature names (assuming these are your features)
        feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                          'Volatility_20', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'Volume_5']

        # Get the weights (assuming model.coef_ is a 2D array, with shape [1, num_features])
        weights = model.coef_[0]  # Assuming the model is SGDClassifier or similar

        # Create the DataFrame for feature weights
        feature_weights = pd.DataFrame({
            'Feature': feature_names, 
            'Weight': weights
        })

        # Sort by absolute weight
        feature_weights['Abs_Weight'] = feature_weights['Weight'].abs()
        feature_weights = feature_weights.sort_values(by='Abs_Weight', ascending=False)

        # Print the feature importance
        print(feature_weights[['Feature', 'Weight']])

        top_feature = feature_weights.iloc[0]  # Get the top feature (row with the highest absolute weight)
        top_feature_name = top_feature['Feature']
        top_feature_weight = top_feature['Weight']

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy for {symbol}: {accuracy:.2f}")

        # Get the last 5 predictions and their actual values
        last_10_predictions = predictions[-10:]
        last_10_actuals = y_test[-10:].values
        last_10_accuracy = accuracy_score(last_10_actuals, last_10_predictions)


        return {"model": model, "y_test": y_test, "predictions": predictions, "accuracy": accuracy, "X_test": X_test, "top_feature": top_feature_name, "top_weight": top_feature_weight, "last_10_accuracy": last_10_accuracy, "best_cv_f1_score": best_cv_f1_score, "last_10_predictions": last_10_predictions}

    except Exception as e:
        print(f"Error during training and prediction for {symbol}: {e}")
        return None
   
def get_account_details(api):
    """
    Fetch and display account details from the trading API.

    Parameters:
        api: The trading API object.

    Returns:
        dict: A dictionary containing account details.
    """
    try:
        # Fetch account details
        account = api.get_account()

        # Display account information
        print("Account Status:", account.status)
        print("Account Buying Power:", account.buying_power)
        print("Equity:", account.equity)
        print("Cash Balance:", account.cash)

        # Return account details as a dictionary
        return {
            "status": account.status,
            "buying_power": account.buying_power,
            "equity": account.equity,
            "cash": account.cash
        }
    except Exception as e:
        print(f"Error fetching account details: {e}")
        return None
   

def predict_and_trade(model, bars_df, api, symbol, quantity, trading_logic):
    """
    Use the model to predict the next direction of the stock and make a trading decision.

    Parameters:
        model (XGBClassifier): The trained model for prediction.
        bars_df (pd.DataFrame): DataFrame containing stock data with features.
        api: API object to execute trades.
        symbol (str): Stock symbol to trade.
        quantity (int): Quantity of stocks to buy or sell.
        trading_logic (function): Function to handle trading logic based on prediction.

    Returns:
        None
    """
    try:
        action = None  # Initialize action variable
        # Get the last row of features for prediction
        next_data = bars_df[['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                               'Volatility_20', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'Volume_5']].iloc[-1:]

        # Make the prediction
        next_prediction = model.predict(next_data)
        # Decide to buy or sell based on the prediction
        if next_prediction[0] == 1:  # Predicting an increase
            trading_logic(api, symbol, action="buy", quantity=quantity)
        elif next_prediction[0] == 0:  # Predicting a decrease
            trading_logic(api, symbol, action="short", quantity=quantity)

    except Exception as e:
        print(f"Error during prediction and trading: {e}")

# Function to get the current price of a stock
def get_current_price (data):
    try:
        # Check if the DataFrame is not empty
        if not data.empty:
            # Extract the last close value
            last_close = data['Close'].iloc[-1]
            last_volume = data['Volume'].iloc[-1]
            last_volume_5 = data['Volume_5'].iloc[-1]
            return last_close, last_volume, last_volume_5
        else:
            print("Data is empty, cannot fetch the last close price.")
            return None
    except KeyError:
        print("The DataFrame does not contain a 'Close' column.")
        return None
    except Exception as e:
        print(f"An error occurred while fetching the last close price: {e}")
        return None


def calculate_quantity(api, symbol, allocation_percent, principal, current_price):
        """
        Calculate the quantity of stock to buy based on a percentage of the principal.
        """
        try:
            # Allocation for this trade
            allocation = 100000 * (allocation_percent / 100)
               
            if current_price is None:
                raise ValueError(f"Could not fetch price for {symbol}")
       
            # Calculate quantity (rounding down to the nearest whole number)
            quantity = int(allocation // current_price)
       
            print(f"Principal: ${principal:.2f}, Allocation: ${allocation:.2f}, Quantity: {quantity}")
            return quantity
        except Exception as e:
            print(f"Error calculating quantity: {e}")
            return 0
   
# Initialize an empty list for storing results
result = []

def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) for the given period.

    Parameters:
        data (pd.DataFrame): DataFrame containing High, Low, and Close columns.
        period (int): Number of periods for ATR calculation.

    Returns:
        float: The ATR value, or None if not enough data.
    """
    if len(data) < period:
        return None  # Not enough data to calculate ATR

    # Calculate True Range (TR)
    data['TR'] = data.apply(
        lambda row: max(
            row['High'] - row['Low'],
            abs(row['High'] - row['Previous_Close']),
            abs(row['Low'] - row['Previous_Close'])
        ),
        axis=1
    )

    # Calculate ATR as the SMA of TR
    atr = data['TR'].rolling(window=period).mean().iloc[-1]
    return atr

def calculate_profit(predictions, y_test, X_test, quantity):
    try:
        # Initialize profit tracking variable
        profit = 0.0

        # Loop through the predictions and actual values to calculate profit/loss
        for i in range(len(y_test) - 1):  # Exclude last value because no next value for comparison
            current_close = X_test[i, 3]  # Access 'Close' value for current row
            next_close = X_test[i + 1, 3]  # Access 'Close' value for next row
           
            # Get the predicted direction for the current interval
            predicted_direction = predictions[i]
           
            # Calculate the difference between predicted and actual closing prices
            if predicted_direction == 1:
                # If predicted 'up' and the next close is higher, it's a profit
                if next_close > current_close:
                    profit += (next_close - current_close) * quantity # Profit = difference
                else:
                    profit -= (current_close - next_close) * quantity # Loss = difference
            elif predicted_direction == 0:
                # If predicted 'down' and the next close is lower, it's a profit
                if next_close < current_close:
                    profit += (current_close - next_close) * quantity # Profit = difference
                else:
                    profit -= (next_close - current_close) * quantity # Loss = difference

        # Output the total profit/loss
        print(f"Total profit/loss from all predictions: ${profit:.2f}")
        return profit
    except Exception as e:
            print(f"Error calculating profit: {e}")

def overall_movement(data):
    # Fetch S&P 500 data
    sp500_data = yf.download('^GSPC', period='500d', interval='1d')

    # Reset multi-level column names (remove 'GSPC' labels if present)
    if isinstance(sp500_data.columns, pd.MultiIndex):
        sp500_data.columns = sp500_data.columns.droplevel(1)

    # Convert index to a column
    sp500_data.reset_index(inplace=True)

    # Rename 'Date' to 'Timestamp' to match the `data` format
    sp500_data.rename(columns={'Date': 'Timestamp'}, inplace=True)

    # Ensure 'Timestamp' in sp500_data matches the format of `data`
    sp500_data['Timestamp'] = pd.to_datetime(sp500_data['Timestamp']).dt.tz_localize(None)  # Remove timezone info
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.tz_localize(None)  # Remove timezone info

    # Calculate daily price movement for S&P 500
    sp500_data['Market_Movement'] = sp500_data['Close'].pct_change().shift(1)

    # Merge with `data` on 'Timestamp' without modifying `data`
    merged_data = data.merge(sp500_data[['Timestamp', 'Market_Movement']], on='Timestamp', how='left')

    # Fill NaN values with 0
    merged_data['Market_Movement'] = merged_data['Market_Movement'].fillna(0)

    return merged_data

def fetch_finnhub_news(symbol, finnhub_api_key, days=200):
    """
    Fetch stock-related news articles from Finnhub API.
    
    Parameters:
        symbol (str): Stock ticker symbol.
        api_key (str): Finnhub API key.
        days (int): Number of past days to fetch news for.
    
    Returns:
        pd.DataFrame: A dataframe with news articles and timestamps.
    """
    # Get the date range
    today = datetime.today()
    from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    # Define the API request URL
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={finnhub_api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching news data:", response.json())
        return None

    news_data = response.json()
    
    # Convert to DataFrame
    news_df = pd.DataFrame(news_data)

    if news_df.empty:
        print("No news articles found.")
        return None

    # Convert timestamp to readable date
    news_df['Timestamp'] = pd.to_datetime(news_df['datetime'], unit='s')
    news_df['Timestamp'] = news_df['Timestamp'].dt.floor('D')
    news_df['Timestamp'] = news_df['Timestamp'].dt.tz_localize('UTC')
    news_df['Timestamp'] = news_df['Timestamp'].dt.tz_convert('America/New_York').dt.normalize()


    
    return news_df[['Timestamp', 'headline', 'summary']]

def analyze_sentiment(news_df):
    """
    Perform sentiment analysis on news headlines.
    
    Parameters:
        news_df (pd.DataFrame): Dataframe containing news headlines.
    
    Returns:
        pd.DataFrame: Updated dataframe with sentiment scores.
    """
    if news_df is None:
        return None

    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(text):
        """Get compound sentiment score from VADER."""
        return analyzer.polarity_scores(text)['compound']

    # Apply sentiment analysis to headlines
    news_df['sentiment']=0.0000
    news_df['sentiment'] = news_df['headline'].apply(get_sentiment)

    return news_df

def aggregate_sentiment(data, news_df):
    """
    Aggregate sentiment scores by date.
    
    Parameters:
        news_df (pd.DataFrame): Dataframe containing sentiment scores.
    
    Returns:
        pd.DataFrame: Aggregated sentiment scores by date.
    """
    if news_df is None:
        return None

    sentiment_by_date = news_df.groupby('Timestamp')['sentiment'].mean().reset_index()
    merged = pd.merge(data, sentiment_by_date, how='left', on='Timestamp')
    merged['sentiment'] = merged['sentiment'].fillna(0)
    tm.sleep(1)  # Pause briefly to avoid API rate limits
    last_row = merged.iloc[[-1]]  # Extract the last row (for later prediction)

    return merged, last_row
   
# Initialize API
api = tradeapi.REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), base_url=BASE_URL)
sp500_symbols = list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'])

# Nasdaq FTP URL for all listed stocks
url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
# Read the data, Nasdaq uses '|' as a separator
df = pd.read_csv(url, sep="|")
# Extract stock symbols (excluding test/delisted stocks)
min_price = 5
Nasdaq_symbols = df[df["Market Category"].isin(["G", "Q"])]["Symbol"].tolist()

NYSE_url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
NYSE_df = pd.read_csv(NYSE_url, sep="|")
NYSE_symbols = NYSE_df[NYSE_df["Exchange"] == "N"]["ACT Symbol"].tolist()

# Initialize variables for multiple stocks
stocks = sp500_symbols # List of stock symbols
stock_status = {symbol: {'holding_stock': False, 'purchase_price': None, 'action': None} for symbol in stocks}
allocation_percent = 5
# Directory to save models (with subdirectories for each stock)
model_dir = "Models_50_4_18_25"
os.makedirs(model_dir, exist_ok=True)
Finnhub_API_Key = "Your API Key Here"


def main_trading_logic():
    # Initialize list to store results
    results = []
    account = api.get_account()
    principal = float(account.cash)
    flipped=False

    for symbol in stocks:
        try:
            print(f"{datetime.now()}: Fetching data and making predictions for {symbol}...")
            bars_df = fetch_historical_data(api, symbol)
            data = add_features(bars_df)
            news_df = fetch_finnhub_news(symbol, Finnhub_API_Key)
            news_df = analyze_sentiment(news_df)
            merged, last_row = aggregate_sentiment(data, news_df)
            #merged = overall_movement(merged)
            result = train_and_predict(merged, symbol)

            if result:  # Check if the result is not None
                model = result.get("model")
                y_test = result.get("y_test")
                predictions = result.get("predictions")
                accuracy = result.get("accuracy")
                X_test = result.get("X_test")
                top_feature = result.get("top_feature")
                top_weight = result.get("top_weight")
                Last_10_accuracy = result.get("last_10_accuracy")
                best_cv_f1_score = result.get("best_cv_f1_score")
                pred_last_10 = result.get("last_10_predictions")

                # Flip predictions if accuracy is 48% or less
                if accuracy <= 0.48:
                    accuracy = 1 - accuracy #Flip accuracy
                    predictions = 1 - predictions  # Flip 0 → 1 and 1 → 0
                    Last_10_accuracy = 1 - Last_10_accuracy
                    pred_last_10 =  1 - pred_last_10
                    flipped = True
                    print(f"Accuracy for {symbol}: {accuracy:.2f}")
                    print(f"Flipped predictions for {symbol} due to low accuracy ({accuracy:.2f})")
                else:
                    flipped=False
            else:
                print(f"Error during training and prediction for {symbol}")


            # Calculate ATR (14)
            atr_14 = calculate_atr(data, period=14)
            current_price, volume, volume_5 = get_current_price(last_row)
            quantity = calculate_quantity(api, symbol, allocation_percent, principal, current_price)

            profit =calculate_profit(predictions, y_test, X_test, quantity)
            profit10 = calculate_profit(pred_last_10, y_test[-10:], X_test[-10:], quantity)
           
            # Append the symbol and accuracy to the results list
            results.append({"Symbol": symbol, "Accuracy": accuracy *100, "Accuracy_10": Last_10_accuracy *100,"Profit": profit, "Profit10": profit10, "Flipped": flipped,"CV_Score": best_cv_f1_score *100, "Volume": volume, "Volume_5": volume_5, "top_feature": top_feature, "top_weight": top_weight, "ATR_14": atr_14,
            "Current Price": round(current_price, 2) if current_price else "N/A"})
           
            # Write results to Excel
            try:
                output_file = "Model_50_4-18-25.xlsx"

                # Read existing data if the file exists
                if os.path.exists(output_file):
                    existing_df = pd.read_excel(output_file)

                    # Append new data and drop duplicates
                    df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
                    df = df.drop_duplicates(subset=["Symbol"], keep="first")  # Keep first occurrence
                else:
                    df = pd.DataFrame(results)

                # Save updated data
                df.to_excel(output_file, index=False)
                print(f"Results saved to {output_file}")
            except Exception as e:
                print(f"Error saving results to Excel: {e}")

            if accuracy >= .54:
                print(f"{datetime.now()}: Accuracy is sufficient ({accuracy:.2f}%) for {symbol}. Proceeding with trade.")
            else:
                print(f"{datetime.now()}: Accuracy is insufficient ({accuracy:.2f}%) for {symbol}. Skipping trade.")

        except Exception as e:
            print(f"Error in trading loop for {symbol}: {e}")

if __name__ == "__main__":
    while True:
        try:
            main_trading_logic()  # Call trading function
            tm.sleep(10000)  # Wait 60 seconds before running again
        except Exception as e:
            print(f"Error in main loop: {e}")
            tm.sleep(60)  # Wait before retrying to avoid rapid failures
