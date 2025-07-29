from pathlib import Path
import pandas as pd
from ..datacode.find_relevant_csv import find_relevant_csv
import numpy as np
from .funcs import portfolio_performance, optimize_variance, optimize_sharpe
import matplotlib.pyplot as plt

root = Path.cwd()
datadir = root/'main'/'data'

portfolio_tickers = ['QQQ', 'SPY', 'EEM', 'SPXS', 'HYG', 'BTC-EUR']
datadirs = [str(datadir/'etfs_1d') for _ in range(5)] + [str(datadir/'crypto_1d')]

ci = None
dfs = []
for dd, pt in zip(datadirs, portfolio_tickers):
    df = find_relevant_csv(dd, pt)
    ci = ci.intersection(df.index) if ci is not None else df.index
    dfs.append(df)

returns = np.empty(shape=(len(ci)-1, len(portfolio_tickers)))
for i, df in enumerate(dfs):
    returns[:,i] = df.loc[ci]['Close'].pct_change().values[1:]

pstds = []
preturns = []

returns_per_asset = np.mean(returns, axis=0) * 365 # annual average return per asset
for sample_return in np.linspace(min(returns_per_asset), max(returns_per_asset), 100):
    optimal_weights = optimize_variance(returns, sample_return)
    total_return, std = portfolio_performance(returns, optimal_weights)
    pstds.append(std)
    preturns.append(total_return)

riskfree = 0.03 # eg. 3%
weights_tangent = optimize_sharpe(returns, returntarget=riskfree)
total_return_tangent, std_tangent = portfolio_performance(returns, weights_tangent)
sharpe_tangent = (total_return_tangent - riskfree) / total_return_tangent

plt.figure(figsize=(10, 6))
plt.plot(pstds, preturns, label="Efficient Frontier", color='blue')

x_vals = np.linspace(0, max(pstds), 100)
cml = riskfree + sharpe_tangent * x_vals
plt.plot(x_vals, cml, label="Capital Market Line (CML)", linestyle='--', color='green')

plt.scatter(std_tangent, total_return_tangent, color='red', label='Tangency Portfolio', zorder=5)

plt.xlabel("Portfolio Risk (Std. Deviation)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.grid(True)
plt.legend()
plt.show()


