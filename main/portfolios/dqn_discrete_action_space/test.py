from stable_baselines3 import DQN
import matplotlib.pyplot as plt
from ...datacode.find_relevant_csv import find_relevant_csv
from .train import PortfolioVenv, actionmapping, portfolio_tickers, result_df
from pathlib import Path
import pandas as pd
import json
import sys
import numpy as np

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

trade_df = None

obss, actions = [], []
weights_taken = []
while not done:
    action, _states = model.predict(obs) # action = integer
    obs, reward, terminated, truncated, _ = env.step(int(action)) # obs = array
    obss.append(obs)
    actions.append(int(action))

    prev_weights = weights_taken[-1] if weights_taken else 0
    weights_taken.append(actionmapping[int(action)])
    cur_weights = weights_taken[-1]
    diff = cur_weights - prev_weights
    bought_sold = diff * env.portfolio_value

    trade_df = pd.concat([trade_df, pd.DataFrame([
        [t, tdel] for t, tdel in zip(portfolio_tickers, bought_sold)
    ], columns=['Ticker', 'BuySellAmount'], index=
        env.df.index[len(env.df)-len(portfolio_tickers):]
        )]) if trade_df is not None else pd.DataFrame([
        [t, tdel] for t, tdel in zip(portfolio_tickers, bought_sold)
    ], columns=['Ticker', 'BuySellAmount'], index=
        env.df.index[len(env.df)-len(portfolio_tickers):]
        )


    portfolio_values.append(env.portfolio_value)
    done = terminated or truncated

buyhold_return = ( # assuming equal weights across assets
    env.df.loc[env.df.index[-1], [f"{t}-Close" for t in portfolio_tickers]].mean()
    /
    env.df.loc[env.df.index[0], [f"{t}-Close" for t in portfolio_tickers]].mean()
    -1
)
std = env.df[[f"{t}-Close" for t in portfolio_tickers]].pct_change().mean(axis=1).std()

this_return = portfolio_values[-1] / portfolio_values[0] - 1
buyhold_returns = env.df[[f"{t}-Close" for t in portfolio_tickers]].mean(axis=1).pct_change().dropna().values
buyhold_returns_cum = np.cumprod(1 + buyhold_returns)

sharpe_ratio = (this_return - 0.03) / std
std_down = buyhold_returns[buyhold_returns < 0].std()
print('filtered buyhold returns:', buyhold_returns[buyhold_returns < 0])
print('std down:'); print(std_down)
sortino_ratio = (this_return - 0.03) / std_down

print('sortino ratio:', sortino_ratio)
print('sharpe ratio:', sharpe_ratio)
print('portfolio return:', this_return)
print('buy and hold return:', buyhold_return)

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj

with open(current/"test_summary.json", "w") as f:
    json.dump(to_serializable({
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
    }), f, indent=2)


fig, ax = plt.subplots()
print('buyhold returns:'); print(buyhold_returns * env.init_cash)
print('port values:'); print(portfolio_values)
ax.plot(buyhold_returns_cum * env.init_cash, label="Buy & Hold")
ax.plot(portfolio_values, label="DQN Strategy")
ax.legend()
plt.title("Portfolio Value Over Time")
plt.show()
plt.savefig(current/'test_graph')
