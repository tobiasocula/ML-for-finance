import gymnasium as gym
import numpy as np
from pathlib import Path
from ...datacode.find_relevant_csv import find_relevant_csv
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

"""
Test implementation of a DQL-'agent' that attempts to determine best action to take
for the portfolio.
"""

class PortfolioVenv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, init_cash, window_size, action_mapping, tickers):
        """
        -df: dataframes with all OHCLV column of all assets. Should have naming convention
        'TICKER-Open', etc.
        -init_cash: initial cash given to the portfolio
        -window_size: amount of rows of df included in observation space
        -action_mapping: dictionary mapping action index to asset weights
        of the portfolio.
        for this experimental example, the actions will represent different
        weight distributions for the assets.
        """
        self.logged_values = []

        self.df = df
        self.tickers = tickers
        self.num_assets = len(df.loc[df.index[0]]) // 5
        self.init_cash = init_cash
        self.nrows = window_size
        self.portfolio_value = init_cash
        self.current_step = window_size
        self.action_mapping = action_mapping

        self.action_space = gym.spaces.Discrete(self.num_assets)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nrows * self.df.shape[1],),
            dtype=np.float32
        )

    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")

    def _get_state(self):
        returns = self.df.pct_change().values
        # returns returns of portfolio (last nrows values)
        return returns[self.current_step - self.nrows : self.current_step].flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Sets the seed internally, optional
        self.current_step = self.nrows
        self.portfolio_value = self.init_cash
        return self._get_state(), {}
    
    def _action_to_weights(self, action):
        return self.action_mapping[action]
    
    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            obs = self._get_state() # or return zeros, or last valid state
            reward = 0
            info = {}
            return obs, reward, terminated, truncated, info

        weights = self._action_to_weights(action)

        returns = self.df.pct_change().iloc[self.current_step][
            [f"{s}-Close" for s in self.tickers]
        ].values

        portfolio_return = np.dot(weights, returns)
        self.portfolio_value *= (1 + portfolio_return)
        self.current_step += 1

        reward = portfolio_return
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        next_state = self._get_state()

        self.logged_values.append({
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "reward": reward,
            "action": action
        })

        return next_state, reward, terminated, truncated, {}





root = Path.cwd()
datadir = root/'main'/'data'
pdir = root/'portfolios'/'test_portfolio_with_dqn'

portfolio_tickers = ['QQQ', 'SPY', 'EEM', 'SPXS', 'HYG', 'BTC-EUR']
datadirs = [str(datadir/'etfs_1d') for _ in range(5)] + [str(datadir/'crypto_1d')]

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
result_df = result_df.loc[result_df.index[:int(learn_pct*len(result_df))]]

actionmapping = {
    0: np.array(6 * [1/6]),
    1: np.array([0.5] + 5 * [0.1]),
    2: np.array([0.1, 0.5] + 4 * [0.1]),
    3: np.array([0.1, 0.1, 0.5] + 3 * [0.1]),
    4: np.array([0.1, 0.1, 0.1, 0.5] + 2 * [0.1]),
    5: np.array(4 * [0.1] + [0.5, 0.1]),
    6: np.array([5 * [0.1] + [0.5]])
}

def main():

    env = PortfolioVenv(result_df, 10_000, 100, actionmapping, portfolio_tickers)

    model = DQN(MlpPolicy, env, verbose=1, learning_rate=1e-3, buffer_size=50000)
    model.learn(total_timesteps=100_000)

    model.save(str(pdir/Path("dqn_portfolio_model")))


if __name__ == '__main__':
    main()