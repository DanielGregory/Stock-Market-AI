import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time as tm
import os
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import torch
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

Finnhub_API_Key = os.getenv("FINNHUB_API_KEY", "")

def fetch_historical_data(symbol, days=500):
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
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data['Timestamp'] = data['Timestamp'].map(pd.Timestamp.timestamp)

        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility_15'] = data['Close'].rolling(window=15).std()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']

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

        data['TR'] = data.apply(
            lambda row: max(
                row['High'] - row['Low'],
                abs(row['High'] - row['Previous_Close']),
                abs(row['Low'] - row['Previous_Close'])
            ),
            axis=1
        )
        data['ATR_14'] = data['TR'].rolling(window=14).mean()

        last_row = data.iloc[[-1]]

        data['Next_Close'] = data['Close'].shift(-1)
        data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)
        data['Direction_5'] = data['Direction'].shift(1).rolling(window=5).mean()

        data.dropna(inplace=True)
        return data, last_row
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame(), pd.DataFrame()

def fetch_finnhub_news(symbol, finnhub_api_key, days=500):
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
    news_df['Timestamp'] = news_df['Timestamp'].map(pd.Timestamp.timestamp)
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
        return data
    sentiment_by_date = news_df.groupby('Timestamp')['sentiment'].mean().reset_index()
    merged = pd.merge(data, sentiment_by_date, how='left', on='Timestamp')
    merged['sentiment'] = merged['sentiment'].fillna(0)
    tm.sleep(1)
    return merged


class StockTradingEnv(gym.Env):
    def __init__(self, symbol, finnhub_api_key=""):
        super(StockTradingEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0: Buy, 1: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.symbol = symbol
        self.finnhub_api_key = finnhub_api_key
        self.data = self.fetch_data()
        self.current_step = 0

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Environment seed set to: {seed}")

    def fetch_data(self):
        bars_df = fetch_historical_data(self.symbol)
        data, last_row = add_features(bars_df)

        if self.finnhub_api_key:
            news_df = fetch_finnhub_news(self.symbol, self.finnhub_api_key)
            news_df = analyze_sentiment(news_df)
            merged = aggregate_sentiment(data, news_df)
        else:
            data['sentiment'] = 0
            merged = data

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                    'Volatility_15', 'Volatility_20', 'Momentum', 'RSI', 'Obv', 'sentiment',
                    'MACD_Crossover', 'ATR_14']
        merged = merged[features]
        return merged

    def reset(self):
        self.current_step = 0
        state = self.data.iloc[self.current_step]
        return state.values

    def step(self, action):
        current_state = self.data.iloc[self.current_step]
        next_state = (
            self.data.iloc[self.current_step + 1]
            if self.current_step + 1 < len(self.data)
            else current_state
        )
        reward, total_trades = self.calculate_profit(action, current_state, next_state)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return next_state.values, reward, done, {}

    @staticmethod
    def get_current_price(data):
        try:
            if not data.empty:
                return data['Close'].iloc[-1]
            print("Data is empty, cannot fetch the last close price.")
            return None
        except KeyError:
            print("The DataFrame does not contain a 'Close' column.")
            return None
        except Exception as e:
            print(f"An error occurred while fetching the last close price: {e}")
            return None

    def calculate_profit(self, action, current_state, next_state):
        current_price = current_state['Close']
        next_price = next_state['Close']
        price_change_percentage = (next_price - current_price) / current_price * 100
        reward = 0
        total_trades = 1
        if action == 0:    # Buy
            reward = price_change_percentage
        elif action == 1:  # Sell
            reward = -price_change_percentage
        return reward, total_trades


try:
    sp500_symbols = list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'])
except Exception:
    sp500_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V"]
    print("Could not fetch S&P 500 list, using default symbols.")

stocks = sp500_symbols
model_dir = "Models_RL"
os.makedirs(model_dir, exist_ok=True)


def main(symbol):
    print(f"Training and evaluating PPO for {symbol}...")
    model_path = os.path.join(model_dir, f"{symbol}_ppo_model.zip")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = StockTradingEnv(symbol, finnhub_api_key=Finnhub_API_Key)
    env = DummyVecEnv([lambda: env])
    env.seed(seed)
    env.action_space.seed(seed)

    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"Loaded saved model for {symbol} from {model_path}")
    else:
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
        model.set_random_seed(seed)
        print(f"No saved model found for {symbol}. Training a new model.")

    model.learn(total_timesteps=8000)
    model.save(model_path)
    print(f"Model for {symbol} saved to {model_path}")

    obs = env.reset()
    done = False
    total_reward = 0
    win_count = 0
    total_trades = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_reward += rewards
        total_trades += 1
        if rewards > 0:
            win_count += 1

    win_rate = win_count / total_trades if total_trades > 0 else 0
    print(f"Total Reward for {symbol}: {total_reward}, Win Rate: {win_rate:.2f}")

    features = {
        'symbol': [symbol],
        'total_reward': [float(total_reward)],
        'win_rate': [win_rate],
        'total_trades': [total_trades]
    }

    output_file = "RL_results.xlsx"
    try:
        df = pd.DataFrame(features)
        if os.path.exists(output_file):
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to Excel: {e}")

    return total_reward


if __name__ == "__main__":
    try:
        for symbol in stocks:
            try:
                main(symbol)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
    except Exception as e:
        print(f"Error in main loop: {e}")
        tm.sleep(60)
