
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
import pandas as pd
import json
import sys
import numpy as np
from ...datacode.find_relevant_csv import find_relevant_csv
from stable_baselines3.sac import MlpPolicy

class PortfolioVenv(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, init_cash, window_size, tickers):
        """
        -df: dataframes with all OHCLV column of all assets. Should have naming convention
        'TICKER-Open', etc.
        -init_cash: initial cash given to the portfolio
        -window_size: amount of rows of df included in observation space
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

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.nrows * self.df.shape[1],),
            dtype=np.float32
        )

    def render(self):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")

    def _get_state(self):
        returns = self.df.pct_change().fillna(0).clip(-1, 1).values[1:, :]
        # returns returns of portfolio (last nrows values)
        return returns[self.current_step - self.nrows : self.current_step].flatten().astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Sets the seed internally, optional
        self.current_step = self.nrows
        self.portfolio_value = self.init_cash
        return self._get_state(), {}
    
    def step(self, action):
        
        if self.current_step >= len(self.df) - 1:
            terminated = True
            truncated = False
            obs = self._get_state() # or return zeros, or last valid state
            reward = 0
            info = {}
            return obs, reward, terminated, truncated, info

        weights = action / action.sum() # action is a np.array with 0 <= value <= 1

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
currentdir = Path(__file__).resolve().parent
etfsdir = root/'main'/'data'/'etfs_1d'
indicesdir = root/'main'/'data'/'indices_1d'
current = root/'portfolios'/'test_portfolio_with_dqn'

portfolio_tickers = []
for f in etfsdir.iterdir():
    portfolio_tickers.append(f.name.split('--')[0])
for f in indicesdir.iterdir():
    portfolio_tickers.append(f.name.split('--')[0])

datadirs = [
    str(etfsdir) for _ in range(len(list(etfsdir.iterdir())))
    ] + [
        str(indicesdir) for _ in range(len(list(indicesdir.iterdir())))
    ]


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


import os
from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
import json

def main():
    checkpoint_path = currentdir / "cont_action_space_model"
    progress_path = currentdir / "training_progress.json"
    segment_timesteps = 20_000
    total_desired_timesteps = 100_000

    # Build environment
    env = PortfolioVenv(result_df, 10_000, 100, portfolio_tickers)

    # Load previous progress if it exists
    if checkpoint_path.with_suffix(".zip").exists():
        print("✅ Resuming from checkpoint")
        model = SAC.load(str(checkpoint_path), env=env)

        if progress_path.exists():
            with open(progress_path, "r") as f:
                progress = json.load(f)
                trained_steps = progress.get("trained_timesteps", 0)
        else:
            trained_steps = 0
    else:
        print("🆕 Starting new training session")
        model = SAC(MlpPolicy, env, verbose=1, learning_rate=1e-3, buffer_size=50000)
        trained_steps = 0

    while trained_steps < total_desired_timesteps:
        steps_to_train = min(segment_timesteps, total_desired_timesteps - trained_steps)
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False)

        trained_steps += steps_to_train

        # Save model
        model.save(str(checkpoint_path))

        # Save progress
        with open(progress_path, "w") as f:
            json.dump({"trained_timesteps": trained_steps}, f)

        print(f"✅ Saved checkpoint after {trained_steps} timesteps.")

if __name__ == '__main__':
    main()
