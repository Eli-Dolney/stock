import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np
import customtkinter as ctk
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tkinter import messagebox, Toplevel

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
    data['RSI'] = RSIIndicator(data['Adj Close']).rsi()
    macd = MACD(data['Adj Close'])
    data['MACD'] = macd.macd()
    data['MACD Signal'] = macd.macd_signal()
    data = data.dropna()

    # Replace infinite values with NaN and then drop them
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
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
def visualize_results(dates, y_test, y_pred, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, y_test, label='Actual Prices', color='blue')
    ax.plot(dates, y_pred, label='Predicted Prices', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

# Visualize actual stock prices
def visualize_actual_stock(data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Adj Close'], label='Actual Prices', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

# Main function with CustomTkinter GUI
def main():
    def fetch_and_predict():
        ticker = ticker_entry.get()
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()

        # Validate date format
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            messagebox.showerror("Invalid Date", "Date format should be YYYY-MM-DD")
            return

        try:
            data = fetch_stock_data(ticker, start_date, end_date)
            if data.empty:
                raise ValueError("No data found for the given dates and ticker symbol.")

            data = preprocess_data(data)

            features = ['Adj Close Lag1', 'Adj Close Lag2', 'MA10', 'MA50', 'Volume', 'RSI', 'MACD', 'MACD Signal']
            X = data[features]
            y = data['Adj Close']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            if len(X_train) == 0 or len(X_test) == 0:
                raise ValueError("Insufficient data to split into training and testing sets.")

            # Train and evaluate Linear Regression model
            lr_model = LinearRegression()
            print("Linear Regression Model")
            lr_model, lr_y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test, lr_model)
            lr_fig = visualize_results(data.index[-len(y_test):], y_test, lr_y_pred, 'Linear Regression Model Predictions')

            # Train and evaluate Random Forest model
            rf_model = RandomForestRegressor()
            print("Random Forest Model")
            rf_model, rf_y_pred = train_and_evaluate_model(X_train, X_test, y_train, y_test, rf_model)
            rf_fig = visualize_results(data.index[-len(y_test):], y_test, rf_y_pred, 'Random Forest Model Predictions')

            # Save the Random Forest model
            joblib.dump(rf_model, 'rf_stock_model.pkl')

            # Predict future stock prices using the Random Forest model
            future_dates = pd.date_range(start=end_date, periods=30, freq='B')
            future_data = data[features].iloc[-2:].copy()

            predicted_prices = []
            for date in future_dates:
                future_data.loc[date] = {
                    'Adj Close Lag1': future_data.iloc[-1]['Adj Close Lag1'],
                    'Adj Close Lag2': future_data.iloc[-1]['Adj Close Lag2'],
                    'MA10': future_data.iloc[-1]['MA10'],
                    'MA50': future_data.iloc[-1]['MA50'],
                    'Volume': future_data.iloc[-1]['Volume'],
                    'RSI': future_data.iloc[-1]['RSI'],
                    'MACD': future_data.iloc[-1]['MACD'],
                    'MACD Signal': future_data.iloc[-1]['MACD Signal']
                }
                future_price = rf_model.predict(pd.DataFrame([future_data.loc[date]]))[0]
                predicted_prices.append(future_price)
                future_data.loc[date, 'Adj Close Lag1'] = future_data.iloc[-1]['Adj Close Lag1']
                future_data.loc[date, 'Adj Close Lag2'] = future_price

            future_fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(future_dates, predicted_prices, label='Predicted Prices', color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Adjusted Close Price')
            ax.set_title('Future Stock Price Predictions')
            ax.legend()
            ax.grid(True)

            # Display charts in the CustomTkinter interface
            for fig in [lr_fig, rf_fig, future_fig]:
                canvas = FigureCanvasTkAgg(fig, master=root)
                canvas.draw()
                canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

            # Open new window for actual stock prices
            new_window = Toplevel(root)
            new_window.title("Actual Stock Prices")

            actual_fig = visualize_actual_stock(data, 'Actual Stock Prices')
            actual_canvas = FigureCanvasTkAgg(actual_fig, master=new_window)
            actual_canvas.draw()
            actual_canvas.get_tk_widget().pack(side=ctk.TOP, fill=ctk.BOTH, expand=1)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # CustomTkinter GUI setup
    ctk.set_appearance_mode("dark")  # Modes: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

    root = ctk.CTk()
    root.title("Stock Price Predictor")

    input_frame = ctk.CTkFrame(root)
    input_frame.pack(pady=20)

    ctk.CTkLabel(input_frame, text="Stock Ticker:").grid(row=0, column=0, padx=10, pady=10)
    ticker_entry = ctk.CTkEntry(input_frame)
    ticker_entry.grid(row=0, column=1, padx=10, pady=10)

    ctk.CTkLabel(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=10)
    start_date_entry = ctk.CTkEntry(input_frame)
    start_date_entry.grid(row=1, column=1, padx=10, pady=10)

    ctk.CTkLabel(input_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=10)
    end_date_entry = ctk.CTkEntry(input_frame)
    end_date_entry.grid(row=2, column=1, padx=10, pady=10)

    predict_button = ctk.CTkButton(input_frame, text="Fetch and Predict", command=fetch_and_predict)
    predict_button.grid(row=3, columnspan=2, pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
