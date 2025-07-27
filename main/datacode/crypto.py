import yfinance as yf
import pandas as pd
import sys
import os
from pathlib import Path

interval = '1h'

root = Path.cwd()
data_dir = root/'main'/'data'/f'crypto_{interval}'

pair = 'BTC-GBP'

df = yf.Ticker(pair).history(interval=interval, period='max')
print(df)
df = df.drop(
    [c for c in df.columns if c not in ['Date', 'Open' ,'High', 'Close', 'Low', 'Volume']], axis=1
)
df = df.tz_localize(None)
name = f"{pair}--{df.index.min()}--{df.index.max()}--{interval}.csv".replace(":", "-")
df.to_csv(os.path.join(data_dir, name))