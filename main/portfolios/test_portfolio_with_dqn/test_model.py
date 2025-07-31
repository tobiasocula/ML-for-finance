from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from ...datacode.find_relevant_csv import find_relevant_csv
from .train import PortfolioVenv, actionmapping, portfolio_tickers, result_df
from pathlib import Path
import pandas as pd
import json


root = Path.cwd()
datadir = root/'main'/'data'
current = Path(__file__).resolve().parent

portfolio_tickers = ['QQQ', 'SPY', 'EEM', 'SPXS', 'HYG', 'BTC-EUR']
datadirs = [str(datadir/'etfs_1d') for _ in range(5)] + [str(datadir/'crypto_1d')]

test_summary = {}

ci = None
dfs = []
for dd, pt in zip(datadirs, portfolio_tickers):
    df = find_relevant_csv(dd, pt)
    ci = ci.intersection(df.index) if ci is not None else df.index
    dfs.append(df.rename({n:f"{pt}-{n}" for n in df.columns}, axis=1))

result_df = None
for df in dfs:
    result_df = pd.concat([result_df, df.loc[ci]], axis=1) if result_df is not None else df.loc[ci]

learn_pct = 0.85
test_df = result_df.loc[result_df.index[int(learn_pct*len(result_df)):]]

model = DQN.load(str(current/Path("dqn")))
env = PortfolioVenv(test_df, init_cash=10_000, window_size=100, 
                    action_mapping=actionmapping, tickers=portfolio_tickers)

obs, _ = env.reset()
done = False
portfolio_values = []

obss, actions = [], []
while not done:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(int(action))
    obss.append(obs)
    actions.append(action)
    portfolio_values.append(env.portfolio_value)
    done = terminated or truncated

weights_taken = [actionmapping[a] for a in actions]

buyhold_return = ( # assuming equal weights across assets
    env.df.loc[env.df.index[-1], [f"{t}-Close" for t in portfolio_tickers]].mean()
    /
    env.df.loc[env.df.index[0], [f"{t}-Close" for t in portfolio_tickers]].mean()
    -1
)
std = env.df[[f"{t}-Close" for t in portfolio_tickers]].pct_change().mean(axis=1).std()

this_return = portfolio_values[0] / portfolio_values[-1] - 1
sharpe_ratio = (this_return - 0.03) / std

print('sharpe ratio:', sharpe_ratio)
print('portfolio return:', this_return)
print('buy and hold return:', buyhold_return)

with open(current/"test_summary.json", "w") as f:
    json.dump({
        "tickers": portfolio_tickers,
        "portfolio return": this_return,
        "sharpe": sharpe_ratio,
        "weights_taken": weights_taken,
        "actions_taken": actions,
        "observations": obss,
        "buy_hold_return": buyhold_return,
        "std_returns": std,
        "init cash": 10_000,
        "window length": 100
    }, f)

plt.plot(portfolio_values)
plt.title("Portfolio Value Over Time")
plt.savefig(current/'test_graph')
