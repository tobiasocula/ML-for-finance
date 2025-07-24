import yfinance as yf
import pandas as pd
import sys
import os

save_dir = 'data/stocks_5m'
interval = '5m'

ticker_data = pd.read_csv('../data/ticker_data.csv')
tickers = ticker_data["Symbol"].values
for t in tickers:

    df = yf.Ticker(t).history(interval=interval, period='max')
    df = df.drop(
        [c for c in df.columns if c not in ['Date', 'Open' ,'High', 'Close', 'Low']], axis=1
    )
    df = df.tz_localize(None)
    name = f"{t} {df.index.min()} {df.index.max()} {interval}.csv"
    name = name.replace(":", "-").replace(" ", "--")
    print(save_dir + '/' + name)
    df.to_csv('../' + save_dir + '/' + name)
        



