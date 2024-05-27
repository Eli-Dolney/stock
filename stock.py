import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

# Fetch historical stock price data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Preprocess the data and add features
def preprocess_data(data):
    data = data.dropna()
    data['MA10'] = data['Adj Close'].rolling(window=10).mean()
    data['MA50'] = data['Adj Close'].rolling(window=50).mean()
    data['Volume'] = data['Volume'].pct_change()
    data['Adj Close Lag1'] = data['Adj Close'].shift(1)
    data['Adj Close Lag2'] = data['Adj Close'].shift(2)
    data = data.dropna()
    return data

# Train and evaluate the model
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\nMSE: {mse}\nMAE: {mae}\nR^2: {r2}")
    return model, y_pred

# Visualize actual vs. predicted prices
def visualize_results(dates, y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, y_test, label='Actual Prices')
    plt.plot(dates, y_pred, label='Predicted Prices')
    plt.legend()
    plt.show()

# Main function
def main():
    ticker = input("Enter the stock ticker symbol: ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    data = fetch_stock_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    features = ['Adj Close Lag1', 'Adj Close Lag2', 'MA10', 'MA50', 'Volume']
    X = data[features]
    y = data['Adj Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train and evaluate Linear Regression model
    lr_model = LinearRegression()
    print("Linear Regression Model")
    lr_model, lr_y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test, lr_model)
    visualize_results(data.index[-len(y_test):], y_test, lr_y_pred)

    # Train and evaluate Random Forest model
    rf_model = RandomForestRegressor()
    print("Random Forest Model")
    rf_model, rf_y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_model)
    visualize_results(data.index[-len(y_test):], y_test, rf_y_pred)

    # Save the Random Forest model
    joblib.dump(rf_model, 'rf_stock_model.pkl')

    # Predict future stock prices using the Random Forest model
    future_dates = pd.date_range(start=end_date, periods=30, freq='B')
    future_data = data[features].iloc[-2:].copy()

    predicted_prices = []
    for date in future_dates:
        future_data.loc[date] = [
            future_data.iloc[-1]['Adj Close Lag1'],
            future_data.iloc[-1]['Adj Close Lag2'],
            future_data.iloc[-1]['MA10'],
            future_data.iloc[-1]['MA50'],
            future_data.iloc[-1]['Volume']
        ]
        future_price = rf_model.predict(future_data.iloc[-1].values.reshape(1, -1))[0]
        predicted_prices.append(future_price)
        future_data.loc[date, 'Adj Close Lag1'] = future_data.iloc[-1]['Adj Close Lag1']
        future_data.loc[date, 'Adj Close Lag2'] = future_price

    plt.figure(figsize=(10, 6))
    plt.plot(future_dates, predicted_prices, label='Predicted Prices')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
