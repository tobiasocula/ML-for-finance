import yfinance as yf
import pandas as pd
import sys
import os

interval = '5m'

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), f'data/indices_{interval}')



indices = ['^GSPC', '^DJI', '^IXIC', '^NYA', '^RUT']
# s&p500, dow jones, nasdaq, nyse, russel



for t in indices:

    df = yf.Ticker(t).history(interval=interval, period='max')
    df = df.drop(
        [c for c in df.columns if c not in ['Date', 'Open' ,'High', 'Close', 'Low']], axis=1
    )
    df = df.tz_localize(None)
    name = f"{t}--{df.index.min()}--{df.index.max()}--{interval}.csv".replace(":", "-")
    df.to_csv(os.path.join(data_dir, name))
        