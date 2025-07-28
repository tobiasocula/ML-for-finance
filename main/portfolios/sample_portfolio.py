import numpy as np, sys
import yfinance as yf
from pathlib import Path
from ..datacode.get_correct_date_range import correct_csvs

from .funcs import *




if __name__ == '__main__':

    root = Path.cwd()
    data_folder = root/'main'/'data'
    ticker_data = data_folder/'stocks_1d'
    btc_data = data_folder/'crypto_1d'
    etfs_data = data_folder/'etfs_1d'

    tickers = ['SPY', 'VGK', 'GLD', 'EEM', 'VWO', 'QQQ', 'BTC-EUR']
    data_dirs = [etfs_data for _ in range(6)] + [btc_data]

    dfs = correct_csvs(tickers, data_dirs)
    returns = np.empty(shape=(len(dfs[0])-1, len(tickers)))
    for i, df in enumerate(dfs):
        returns[:,i] = df['Close'].pct_change().values[1:]

    print('RETURNS:'); print(returns)

    cov = np.cov(returns, rowvar=False)
    corr = np.corrcoef(returns, rowvar=False)
    L, M = np.linalg.eig(cov)
    D = np.diag(L)

    print('cov matrix:'); print(cov)
    print('M'); print(M)
    print('eigenvals'); print(L)

    # cov = MDM^T

    ENB1 = sum(L)**2 / sum(k**2 for k in L)
    print('ENB1:', ENB1)

    # find optimal weights for variance minimalization
    w_variance = optimize_variance(returns)

    # find optimal weights to maximize sortino
    w_sortino, _ = optimize_sortino(returns)

    print('variance weights:'); print(w_variance)
    print('sortino weights:'); print(w_sortino)

    # calculate ENB2
    # for both sets of weights
    print('ENB2 for variance:'); print(ENB2(w_variance, L, M))
    print('ENB2 for sortino:'); print(ENB2(w_sortino, L, M))
