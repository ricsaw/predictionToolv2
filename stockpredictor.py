import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from newsapi import NewsApiClient

# Initialize the News API client
news_api_key = 'c798e67f77814d1dbddfd8ee5af67362'  # Replace with your actual News API key
newsapi = NewsApiClient(api_key=news_api_key)


# Function to fetch stock data
def fetch_stock_data(ticker, period='1y'):
    try:
        stock_data = yf.download(ticker, period=period)
        if stock_data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None


# Function to fetch real-time stock price
def fetch_real_time_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        real_time_data = stock.history(period='1d')
        return real_time_data['Close'].iloc[-1] if not real_time_data.empty else None
    except Exception as e:
        print(f"Error fetching real-time price: {e}")
        return None


# Function to fetch news articles for a given ticker
def fetch_news_articles(ticker):
    try:
        articles = newsapi.get_top_headlines(q=ticker, language='en', page_size=100)
        return articles['articles']
    except Exception as e:
        print(f"Error fetching news articles: {e}")
        return []


# Function to calculate sentiment scores from articles using VADER
def calculate_sentiment(news_data):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(news['title'] + " " + (news.get('description', '')))['compound']
                  for news in news_data]
    return np.mean(sentiments) if sentiments else 0


# Function to create features for prediction
def create_features(stock_data, sentiment_score):
    stock_data = stock_data.copy()
    stock_data['Date'] = stock_data.index
    stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days
    stock_data['Lagged Price'] = stock_data['Adj Close'].shift(1)
    stock_data['Price Change'] = stock_data['Adj Close'].pct_change()
    stock_data['Volume Change'] = stock_data['Volume'].pct_change()
    stock_data['Sentiment Score'] = sentiment_score
    return stock_data.dropna()


# Function to perform cross-validation and predict future prices using Random Forest
def predict_future_prices(stock_data, days=30):
    X = stock_data[['Days', 'Lagged Price', 'Price Change', 'Volume Change', 'Sentiment Score']]
    y = stock_data['Adj Close']

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Cross-validation MSE: {mean_squared_error(y_test, y_pred):.4f}")

    future_days = np.array(range(stock_data['Days'].max() + 1, stock_data['Days'].max() + 1 + days)).reshape(-1, 1)
    future_prices = []

    last_row = stock_data.iloc[-1]
    for day in range(days):
        new_data = np.array([[last_row['Days'] + 1, last_row['Adj Close'], last_row['Price Change'],
                              last_row['Volume Change'], last_row['Sentiment Score']]])
        next_price = model.predict(new_data)[0]
        future_prices.append(next_price)

        # Update the last row for the next iteration
        last_row['Days'] += 1
        last_row['Lagged Price'] = last_row['Adj Close']
        last_row['Price Change'] = (next_price - last_row['Adj Close']) / last_row['Adj Close']
        last_row['Adj Close'] = next_price
        last_row['Volume Change'] = last_row['Volume'] * (1 + np.random.normal(0, 0.01))  # Simulate volume change

    return future_days, future_prices


# Function to plot the data
def plot_data(stock_data, future_days, future_prices, ticker):
    plt.figure(figsize=(14, 10))

    future_dates = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), periods=len(future_prices))

    combined_data = pd.DataFrame({
        'Date': stock_data.index.tolist() + future_dates.tolist(),
        'Adj Close': stock_data['Adj Close'].tolist() + future_prices
    })

    # Plot historical and predicted prices
    plt.subplot(2, 1, 1)
    plt.plot(combined_data['Date'], combined_data['Adj Close'], label='Price', color='blue')
    plt.axvline(x=stock_data.index[-1], color='red', linestyle='--', label='Prediction Start')
    plt.title(f'{ticker} Historical Prices and Future Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xlim([combined_data['Date'].min(), combined_data['Date'].max()])
    plt.legend()

    # Plot future predictions
    plt.subplot(2, 1, 2)
    plt.bar(future_dates, future_prices, color='orange', alpha=0.7)
    plt.title(f'{ticker} Future Price Predictions (Bar Graph)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')

    plt.tight_layout()
    plt.show()

    print("\nPredicted future prices:")
    for date, price in zip(future_dates, future_prices):
        print(f"Date: {date.date()}, Predicted Price: ${price:.2f}")


# Main execution
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., AAPL): ").strip()
    days_in_future = int(input("Enter number of days to predict: ").strip())

    stock_data = fetch_stock_data(ticker)
    if stock_data is not None:
        news_data = fetch_news_articles(ticker)  # Fetch news articles
        sentiment_score = calculate_sentiment(news_data)  # Calculate average sentiment
        stock_data = create_features(stock_data, sentiment_score)

        future_days, future_prices = predict_future_prices(stock_data, days_in_future)
        plot_data(stock_data, future_days, future_prices, ticker)

        # Fetch and display real-time stock price
        real_time_price = fetch_real_time_price(ticker)
        if real_time_price is not None:
            print(f"\nReal-time price of {ticker}: ${real_time_price:.2f}")
