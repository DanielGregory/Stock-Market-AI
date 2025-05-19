#import the libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import alpaca_trade_api as tradeapi
import time as tm
import os
import tensorflow as tf
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import torch


#Alpaca API key
os.environ["APCA_API_KEY_ID"] = "YourAPIKey"
os.environ["APCA_API_SECRET_KEY"] = "YourSecretKey"
Finnhub_API_Key = "YOUR_FINNHUB"

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
        start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%dT%H:%M:%SZ')

        # Fetch bars using the specified timeframe
        bars = api.get_bars(symbol, tradeapi.TimeFrame.Day, start=start_date, end=end_date, limit=10000)

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
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        data['Timestamp'] = data['Timestamp'].map(pd.Timestamp.timestamp)

       
        # Add additional features
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['Volatility_15'] = data['Close'].rolling(window=15).std()
        data['Volatility_20'] = data['Close'].rolling(window=20).std()
        data['Momentum'] = data['Close'] - data['Previous_Close']

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

        last_row = data.iloc[[-1]]
        
        # Create the Direction target
        data['Next_Close'] = data['Close'].shift(-1)
        data['Direction'] = (data['Next_Close'] > data['Close']).astype(int)  # 1 if increase, 0 if decrease
        data['Direction_5'] = data['Direction'].shift(1).rolling(window=5).mean()


        # Drop any rows with missing values
        data.dropna(inplace=True)

        return data, last_row
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        return pd.DataFrame()
    
def fetch_finnhub_news(symbol, finnhub_api_key, days=500):
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
    news_df['Timestamp'] = news_df['Timestamp'].map(pd.Timestamp.timestamp)


    
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

    return merged
   
class StockTradingEnv(gym.Env):
    def __init__(self, symbol, api, Finnhub_API_Key):
        super(StockTradingEnv, self).__init__()

        # Define action space (Buy, Sell)
        self.action_space = spaces.Discrete(2)  # 0: Buy, 1: Sell
        
        # Observation space: Assuming you have 10 features in your dataset
        # Adjust this to match the shape of your data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.symbol = symbol
        self.api = api
        self.Finnhub_API_Key = Finnhub_API_Key

        # Fetch and process data (state)
        self.data = self.fetch_data()

        self.current_step = 0

    def seed(self, seed=None):
        """
        Set the random seed for reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Environment seed set to: {seed}")


    def fetch_data(self):
        """
        Fetch data (bars and sentiment) and return merged dataset.
        """
        bars_df = fetch_historical_data(self.api, self.symbol)
        data, last_row = add_features(bars_df)
        news_df = fetch_finnhub_news(self.symbol, self.Finnhub_API_Key)
        news_df = analyze_sentiment(news_df)
        merged = aggregate_sentiment(data, news_df)
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Previous_Close', 'MA_20',
                    'Volatility_15', 'Volatility_20', 'Momentum', 'RSI', 'Obv', 'sentiment', 'MACD_Crossover', 'ATR_14']
        merged = merged[features] 
        return merged

    def reset(self):
        """
        Resets the environment, returns the initial state.
        """
        self.current_step = 0
        state = self.data.iloc[self.current_step]
        self.result = []
        return state.values  # Return as numpy array (shape should match the observation space)

    def step(self, action):
        """
        Takes an action and returns the next state, reward, and whether the episode is done.
        """
        current_state = self.data.iloc[self.current_step]
        
        # Take action and calculate reward (profit/loss)
        next_state = self.data.iloc[self.current_step + 1] if self.current_step + 1 < len(self.data) else current_state
        reward, total_trades = self.calculate_profit(action, current_state, next_state)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1  # End condition: end of data

        return next_state.values, reward,  done, {}
    
    # Function to get the current price of a stock
    def get_current_price (data):
        try:
            # Check if the DataFrame is not empty
            if not data.empty:
                # Extract the last close value
                last_close = data['Close'].iloc[-1]
                return last_close
            else:
                print("Data is empty, cannot fetch the last close price.")
                return None
        except KeyError:
            print("The DataFrame does not contain a 'Close' column.")
            return None
        except Exception as e:
            print(f"An error occurred while fetching the last close price: {e}")
            return None

    def calculate_profit(self, action, current_state, next_state):
        """
        Calculate the profit/loss from the action taken and the change in stock price, 
        normalized by the percentage change to make rewards comparable across different stocks.
        """
        current_price = current_state['Close']  # Current price (from observation)
        next_price = next_state['Close']  # Next price (from observation)
        
        # Calculate the percentage change in the stock price
        price_change_percentage = (next_price - current_price) / current_price * 100

        # Reward calculation logic based on the action
        reward = 0
        wins = 0
        losses = 0
        total_trades =0
        
        if action == 0:  # Buy action
            reward = price_change_percentage  # Positive reward if price goes up
        elif action == 1:  # Sell action
            reward = -price_change_percentage  # Negative reward if price goes down

        total_trades += 1

        return reward, total_trades
    
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
        

   
# Initialize API
api = tradeapi.REST(os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY"), base_url=BASE_URL)
sp500_symbols = list(pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]['Symbol'])

# Nasdaq FTP URL for all listed stocks
url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
# Read the data, Nasdaq uses '|' as a separator
df = pd.read_csv(url, sep="|")
# Extract stock symbols (excluding test/delisted stocks)
Nasdaq_symbols = df[df["Test Issue"] == "N"]["Symbol"].tolist()

# Initialize variables for multiple stocks
stocks = sp500_symbols  # List of stock symbols
stock_status = {symbol: {'holding_stock': False, 'purchase_price': None, 'action': None} for symbol in stocks}
allocation_percent = 5

# Directory to save models (with subdirectories for each stock)
model_dir = "Sp500_models_RL_4_16_25"
os.makedirs(model_dir, exist_ok=True)


def main(symbol):
    print(f"Training and evaluating PPO for {symbol}...")
    # Define subdirectory for the stock symbol
    model_path = os.path.join(model_dir, f"{symbol}_ppo_model.zip")  # Save the model directly in model_dir


    env = StockTradingEnv(symbol, api, Finnhub_API_Key)
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv

    seed = 42  # Choose any number

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set the seed in the environment
    env.seed(seed)
    env.action_space.seed(seed)

     # Check if a saved model exists
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)  # Load the model and attach the environment
        print(f"Loaded saved model for {symbol} from {model_path}")
    else:
        # Initialize a new model if no saved model exists
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
        model.set_random_seed(seed)
        print(f"No saved model found for {symbol}. Training a new model.")


    # Train the model
    model.learn(total_timesteps=8000)  # Train the agent

    # Save the model after training
    model.save(model_path)
    print(f"Model for {symbol} saved to {model_path}")

    # Evaluate the agent
    obs = env.reset()
    done = False
    total_reward = 0
    win_rate = 0
    win_count = 0
    total_trades = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        total_reward += rewards
        total_trades += 1
        if rewards > 0:
            win_count += 1
    win_rate = win_count / total_trades
    print(f"Total Reward for {symbol}: {total_reward}")

    # Collect features (add more features as needed)
    features = {
        'symbol': symbol,
        'total_reward': total_reward,
        'win_rate': win_rate,
        'total_trades': total_trades
        # Add more features like model performance, rewards, etc.
    }

    # Write results to Excel
    output_file = "S&P500_RL_4-16-25.xlsx"
    try:
        # Convert features to a DataFrame
        df = pd.DataFrame(features)

        # Check if the file exists
        if os.path.exists(output_file):
            # Load the existing workbook and append the data
            with pd.ExcelWriter(output_file, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
        else:
            # Create a new file and write the data
            df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to Excel: {e}")

    return total_reward #return the total reward to be used in the main loop.

if __name__ == "__main__":
    try:
        for symbol in stocks:
            try:
                total_rewards = main(symbol)  # Call trading function for each stock.
            except Exception as e:
                print(f"Error processing {symbol}: {e}")  # Log the error and continue
    except Exception as e:
        print(f"Error in main loop: {e}")
        tm.sleep(60)  # Wait before retrying to avoid rapid failures
