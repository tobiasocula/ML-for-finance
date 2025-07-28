from .funcs import random_portfolios, optimize_sortino, optimize_variance, ENB2
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


ps = random_portfolios(tickers, subset_size=5, datafolders=folders, attempts=200,
                       optimize_funcs=[
                           optimize_variance, optimize_variance
                       ], eval_funcs=[
                           ENB2
                       ]
)

for port in ps:
    print('tickers:', port['tickers'])
    print('weights:', port['allweights'])
    print('metrics:', port['allmetrics'])
