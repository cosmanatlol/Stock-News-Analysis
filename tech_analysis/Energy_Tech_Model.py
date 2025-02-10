#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Preprocessing and feature extraction
def preprocess(stock_symbols, sentiment_df):
    strat = ta.Strategy(
        name='Best Strategy Ever',
        ta=[
            {'kind': 'ema', 'length': 10, 'col_names': 'ema_10'},
            {'kind': 'ema', 'length': 25, 'col_names': 'ema_25'},
            {'kind': 'hma', 'length': 50, 'col_names': 'hma_50'},
            {'kind': 'rsi', 'col_names': 'rsi'},
            {'kind': 'macd', 'col_names': ('macd', 'macd_h', 'macd_s')},
            {'kind': 'bbands', 'std': 1, 'col_names': ('BBL', 'BBM', 'BBU', 'BBB', 'BBP')},
        ]
    )

    stock_data = []
    for symbol in stock_symbols:
        sentiment_for_stock = sentiment_df[sentiment_df['stock'] == symbol]
        min_date_with_rating = sentiment_for_stock['date'].min()
        if pd.isna(min_date_with_rating):
            print(f"No sentiment data for {symbol}. Skipping...")
            continue

        # Adjust the end date to the last date with sentiment data (January 31, 2025)
        max_date_with_rating = sentiment_for_stock['date'].max()
        
        data = yf.download(symbol, start=min_date_with_rating, end=max_date_with_rating).reset_index()
        data.ta.strategy(strat)
        data['pct_change'] = data['Close'].pct_change()
        data['target'] = data['Close'].shift(-1)
        data['Symbol'] = symbol
        
        # Convert 'Date' column to datetime for merging
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Merge with sentiment data (make sure 'date' column in sentiment data is also datetime)
        sentiment_for_stock['date'] = pd.to_datetime(sentiment_for_stock['date'])
        
        # Merge data with sentiment ratings
        merged_data = pd.merge(data, sentiment_for_stock, how='inner', left_on='Date', right_on='date')
        
        # Fill missing ratings with 0
        merged_data['rating'].fillna(0, inplace=True)
        
        stock_data.append(merged_data.dropna())

    return pd.concat(stock_data, ignore_index=True)


def predict_next_days_for_stock(model, last_stock_data, days=5):
    # Repeat the last row for 'days' times
    future_features = pd.concat([last_stock_data.tail(1)] * days, ignore_index=True)
    
    # Remove non-numeric columns (e.g., stock, date)
    numeric_features = future_features.select_dtypes(include=[np.number])
    
    # Predict future values
    future_predictions = model.predict(numeric_features)
    return future_predictions

# Main workflow
if __name__ == "__main__":
    # Load sentiment data
    sentiment_data = pd.read_csv("sentiment_data.csv")  # Ensure it contains 'stock', 'date', and 'rating'
    
    # Convert the 'date' column to full date format
    sentiment_data['date'] = sentiment_data['date'].apply(lambda x: pd.to_datetime(f"2025-01-{int(x):02d}"))

    # Define energy stocks
    energy_stocks_test = [
    "WMB",  # The Williams Companies Inc
    "KMI",  # Kinder Morgan Inc
    "OKE",  # ONEOK Inc
    "OXY",  # Occidental Petroleum Corp
    "CVX",  # Chevron Corp
    "XOM",  # Exxon Mobil Corp
    "PSX",  # Phillips 66
    "EOG",  # EOG Resources Inc
    "FANG", # Diamondback Energy Inc
    "SLB"   # Schlumberger NV
    ]

    # Preprocess data    
    data = preprocess(energy_stocks_test, sentiment_df=sentiment_data)
    
    # Prepare features and target
    features = data.drop(columns=['Date', 'Close', 'target', 'Symbol', 'date', "stock"])
    target = data['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)

    # Train XGBoost model
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
    xg_reg.fit(X_train, y_train)

    # Make predictions
    predictions = xg_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    r2 = r2_score(y_test, predictions)

    # Plot feature importance
    #xgb.plot_importance(xg_reg)
    #plt.show()

    # Predict next 5 days for all stocks
    all_predictions = {}
    for stock in energy_stocks_test:
        print(f"\nPredicting for {stock}...")

        # Filter data for the specific stock
        stock_data = data[data['Symbol'] == stock]
        stock_features = stock_data.drop(columns=['Date', 'Close', 'target', 'Symbol', 'date'])

        # Predict for the next 5 days
        future_predictions = predict_next_days_for_stock(xg_reg, stock_features, days=5)
        all_predictions[stock] = future_predictions

    print("\nPredictions for the next 5 days (after January 31, 2025):")
    for stock, pred in all_predictions.items():
        print(f"{stock}: {pred}")

# Load actual data for the next 5 days
actual_data = {}
for stock in energy_stocks_test:
    actual = yf.download(stock, start="2025-02-01", end="2025-02-08").reset_index()
    actual_data[stock] = actual['Open'].values[:5]

# Calculate predictions, actuals, and percent difference
average_percent_diff = {}

for stock, pred in all_predictions.items():
    actual = actual_data.get(stock, [None] * 5)
    
    # Calculate percent difference for each day
    percent_diff = [
        abs((p - a) / a) * 100 if a is not None and a != 0 else None
        for p, a in zip(pred, actual)
    ]
    
    # Compute average percent difference, ignoring None values
    avg_diff = np.nanmean([d for d in percent_diff if d is not None])
    average_percent_diff[stock] = avg_diff

# Print overall summary
print("\nAverage Percent Difference for Each Stock:")
for stock, avg_diff in average_percent_diff.items():
    print(f"{stock}: {avg_diff:.2f}%")

