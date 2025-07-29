from .funcs import random_portfolios, optimize_sortino, optimize_variance, ENB2
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd

root = Path.cwd()
datadir = root/'main'/'data'
etfsdir = datadir/'etfs_1d'
cryptodir = datadir/'crypto_1d'
# stocksdir = datadir/'stocks_1d'

tickers = []
folders = []

dfs = []
cidx = None
for dir in [etfsdir, cryptodir]:
    for file in dir.iterdir():
        dfs.append(pd.read_csv(str(dir/file)))
        tickers.append(str(file).split('--')[0])
for df in dfs:
    if cidx is None:
        cidx = df.index
    else:
        cidx = cidx.intersection(df.index)

dfs = [df.loc[cidx] for df in dfs]

ps = random_portfolios(tickers, subset_size=5, dfs=dfs, attempts=200,
                       optimize_funcs=[
                           optimize_variance, optimize_variance, optimize_sortino
                       ], eval_funcs=[
                           ENB2
                       ]
)

for port in ps:
    print('tickers:', port['tickers'])
    print('weights:', port['allweights'])
    print('metrics:', port['allmetrics'])
