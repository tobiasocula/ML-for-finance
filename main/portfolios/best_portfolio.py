from .funcs import optimize_portfolio, optimize_sortino, optimize_variance, ENB2, risk_parity, risk_parity_loss
from pathlib import Path
import sys
import pandas as pd

root = Path.cwd()
datadir = root/'main'/'data'
etfsdir = datadir/'etfs_1d'
cryptodir = datadir/'crypto_1d'

tickers = []
dfs = []
cidx = None
for dir in [etfsdir, cryptodir]:
    for file in dir.iterdir():
        dfs.append(pd.read_csv(str(dir/file)))
        tickers.append(str(file.name).split('--')[0])
for df in dfs:
    if cidx is None:
        cidx = df.index
    else:
        cidx = cidx.intersection(df.index)

dfs = [df.loc[cidx] for df in dfs]

best_tickers, best_weights, best_metric = optimize_portfolio(tickers, subset_size=8, dfs=dfs,
                        attempts=20, optimize_func=risk_parity, eval_func=risk_parity_loss)

print('tickers:'); print(best_tickers)
print('best weights:'); print(best_weights)
print('metric:', best_metric)