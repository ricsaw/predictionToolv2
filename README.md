# Stock Price Prediction with Sentiment Analysis

## Overview
This project uses historical stock data and sentiment analysis from financial news to predict future stock prices using a Random Forest regression model. It integrates various libraries to fetch real-time stock prices, retrieve relevant news articles, perform sentiment analysis, and visualize predictions.

## Features
- Fetches historical stock data using Yahoo Finance (`yfinance`).
- Retrieves real-time stock prices.
- Gathers news articles related to the stock using NewsAPI.
- Performs sentiment analysis on news articles using VADER Sentiment Analyzer.
- Extracts features from stock data for predictive modeling.
- Trains a `RandomForestRegressor` model with cross-validation.
- Predicts future stock prices for a given number of days.
- Visualizes historical and predicted stock prices.

## Installation
To run this project, install the required dependencies:

```sh
pip install numpy pandas yfinance matplotlib scikit-learn vaderSentiment newsapi-python
```

## Usage
1. Run the script:
   ```sh
   python script.py
   ```
2. Enter the stock ticker symbol when prompted (e.g., `AAPL`).
3. Specify the number of days to predict.
4. The script will:
   - Fetch stock data and news articles.
   - Analyze sentiment from news headlines.
   - Train a model and predict future stock prices.
   - Display real-time stock price.
   - Plot historical and predicted stock prices.

## Configuration
- Replace `news_api_key` with your actual NewsAPI key.

## Dependencies
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `yfinance` - Stock data retrieval
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning
- `vaderSentiment` - Sentiment analysis
- `newsapi-python` - News API client

## Output
- Real-time stock price
- Predicted stock prices for the specified future days
- Graphical visualization of historical and predicted prices

## Notes
- Ensure that your NewsAPI key is valid and has sufficient quota.
- The model does not guarantee accurate predictions but provides insights based on historical trends and sentiment analysis.

## License
This project is for educational purposes only. Use at your own risk.

