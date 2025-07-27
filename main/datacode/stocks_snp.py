import yfinance as yf
import pandas as pd
import sys
import os
from pathlib import Path

interval = '1d'

root = Path.cwd()
data_dir = root/'main'/'data'/f'stocks_{interval}'
ticker_data_path = root/'main'/'data'

ticker_data = pd.read_csv(ticker_data_path/'ticker_data.csv')
tickers = ticker_data["Symbol"].values
for t in tickers:

    df = yf.Ticker(t).history(interval=interval, period='max')
    df = df.drop(
        [c for c in df.columns if c not in ['Date', 'Open' ,'High', 'Close', 'Low', 'Volume']], axis=1
    )
    df = df.tz_localize(None)
    name = f"{t} {df.index.min()} {df.index.max()} {interval}.csv"
    name = name.replace(":", "-").replace(" ", "--")
    df.to_csv(os.path.join(data_dir, name))
        



