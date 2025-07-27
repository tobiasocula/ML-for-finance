import numpy as np
import yfinance as yf

tickers = ['AAPL', 'MSFT', 'GOOG']

data = yf.download(tickers, interval='1d', period='max')
print(data)