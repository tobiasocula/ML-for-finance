import yfinance as yf
from pathlib import Path

root = Path.cwd()

interval = '5m'
tickers = [
    "TSLL", "TSLZ", "SOXS", "SOXL", "SQQQ", "TSLQ", "SPY", "IBIT", "TSLS", "ETHA",
    "TQQQ", "SPXS", "MSTZ", "AMDL", "QQQ", "HYG", "XLF", "TLT", "LQD", "ULTY",
    "GDX", "SLV", "IWM", "MSTU", "EWZ"
]

save_dir = root/'main'/'data'/f'etfs_{interval}'

for t in tickers:

    df = yf.Ticker(t).history(interval=interval, period='max')
    df = df.drop(
        [c for c in df.columns if c not in ['Date', 'Open' ,'High', 'Close', 'Low', 'Volume']], axis=1
    )
    df = df.tz_localize(None)
    name = f"{t} {df.index.min()} {df.index.max()} {interval}.csv"
    name = name.replace(":", "-").replace(" ", "--")
    df.to_csv(save_dir/name)