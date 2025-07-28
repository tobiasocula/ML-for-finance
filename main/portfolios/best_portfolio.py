from .funcs import optimize_portfolio, optimize_sortino, optimize_variance, ENB2
from pathlib import Path
import sys

root = Path.cwd()
datadir = root/'main'/'data'
etfsdir = datadir/'etfs_1d'
cryptodir = datadir/'crypto_1d'

tickers = []
folders = []

for dir in [etfsdir, cryptodir]:
    for file in dir.iterdir():
        ticker = file.name.split('--')[0]
        tickers.append(ticker)
        folders.append(str(dir))

best_tickers, best_weights, best_metric = optimize_portfolio(tickers, subset_size=8, datafolders=folders,
                        attempts=20, optimize_func=optimize_sortino, eval_func=ENB2)

print('tickers:'); print(best_tickers)
print('best weights:'); print(best_weights)
print('metric:', best_metric)