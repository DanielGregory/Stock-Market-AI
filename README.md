# Stock Market AI 🧠📈

This repository contains experimental AI models for predicting stock market trends and simulating trading decisions. The focus is on building intelligent systems that can analyze financial data and evaluate strategies through machine learning.

## 🔧 Models Included

### `AI_Test/` – SGDClassifier Model
A baseline model using Scikit-learn’s `SGDClassifier` to perform simple price direction classification (up/down) based on historical features.

### `RL_Model/` – Reinforcement Learning Model
A custom Reinforcement Learning environment where an agent learns to trade stocks through reward-based learning using historical market data.

### `RNN_Model/` – Recurrent Neural Network
(Currently in development/testing) A time-series model that learns from sequences of stock data to make future price predictions and simulate trading behavior.

## 💾 Outputs

Each model:
- **Saves its trained model** (e.g., `.pkl`) for later use or testing.
- **Exports results** (e.g., predictions, profits, accuracy, etc.) into Excel (`.xlsx`) files for easy inspection and visualization.

## 🔑 API Keys Required

This project requires access to the following APIs:

- **Alpaca** – for fetching real-time stock data and (optional) paper trading execution.
- **Finnhub** – for supplemental market data like financial metrics, technical indicators, and news.
