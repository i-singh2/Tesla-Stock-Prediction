# This script is responsible for predicting the day's closing price of Tesla stock before the market opens 
# and recommending whether to buy, sell, or hold the stock based on the prediction.

import yfinance as yf
import pandas as pd
import ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, message="X does not have valid feature names"
)


# --- Functions ---
# Fetch Tesla stock data from Yahoo Finance
def fetch_tesla_data(start_date: str, end_date: str):

    print(f"\nFetching Data...")

    # Download stock data from start date to yesterday
    tesla_data = yf.download("TSLA", start=start_date, end=end_date, interval="1d")

    # Reset index to make 'Date' a column
    tesla_data.reset_index(inplace=True)

    # Keep only the necessary columns
    tesla_data = tesla_data[["Date", "Open", "High", "Low", "Close", "Volume"]]

    # Save the cleaned data for backup
    tesla_data.to_csv("tesla_stock_data.csv", index=False)

    print("Data saved as 'tesla_stock_data.csv'")
    return tesla_data

# Calculate technical indicators for the entire DataFrame.
def calculate_features(df):
    df["SMA_5"] = ta.trend.sma_indicator(df["Close"], window=5)
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["Momentum"] = df["Close"].diff()
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Volatility"] = df["Close"].rolling(10).std()
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    return df


# Returns the next trading day and skips weekends
def next_day(current_date):

    new_date = current_date + timedelta(days=1)
    while new_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        new_date += timedelta(days=1)
    return new_date


# --- Model Development ---
# Prompt the user for the current date
current_date_str = input("Enter today's date <YYYY-MM-DD>: ")
current_date = datetime.strptime(current_date_str, "%Y-%m-%d")

# Fetch and save the cleaned data
df = fetch_tesla_data("2015-01-01", current_date_str)

# Load the cleaned Tesla stock data from the CSV file
df = pd.read_csv("tesla_stock_data.csv")

# Convert the 'Date' column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Convert price and volume columns to numeric data types
columns_to_convert = ["Open", "High", "Low", "Close", "Volume"]
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values
df.dropna(inplace=True)

# Calculate technical indicators to be used as features
df = calculate_features(df)

# Drop any NaN values generated from rolling calculations
df.dropna(inplace=True)

# Define the target variable, which is next day's closing price
df["Target_Price"] = df["Close"].shift(-1)
# Use linear interpolation to fill the last missing value so that the latest day's data is not dropped
df["Target_Price"] = df["Target_Price"].interpolate(method="linear")

# List all the features to be used in the model
features = [
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "EMA_10",
    "RSI_14",
    "MACD",
    "Momentum",
    "Daily_Return",
    "Volatility",
    "ADX",
]

# Select features (X) and target variable (y)
X = df[features]
y = df["Target_Price"]

# Perform a time-based train-test split (first 80% for training, last 20% for testing)
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Create a pipeline with scaling and a regression model
pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])

# Train the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}\n")

# --- Forecasting for the next day ---
# Extract the feature vector from the last row of df
last_features = df.iloc[-1][features].values.reshape(1, -1)

# Predict the next day's closing price using the trained model (pipeline)
predicted_price = pipeline.predict(last_features)[0]

# Determine the new trading day based on the last date in df
last_date = df.iloc[-1]["Date"]
new_date = next_day(last_date)

# Determine the action based on the predicted price and the previous day's closing price
prev_price = df.iloc[-1]["Close"]
if predicted_price > prev_price:
    action = "BUY (HOLD if you already have shares)"  # Buy or hold if the price is predicted to increase
else:
    action = "SELL"  # Sell if the price is predicted to decrease

# Print the previous day's closing price and date
print("\nPrevious Day's Summary:")
print(f"Date: {last_date.strftime('%Y-%m-%d')}")
print(f"Closing Price: ${prev_price:.2f}")

# Save the forecast result for this day
forecast_result = {
    "Date": new_date.strftime("%Y-%m-%d"),
    "Predicted Price": f"${predicted_price:.2f}",
    "Action": action,
}

# Output
print("\nForecast for today:")
print(f"Date: {forecast_result['Date']}")
print(f"Predicted Closing Price: {forecast_result['Predicted Price']}")
print(f"Recommended Action: {forecast_result['Action']}\n")
