import gymnasium as gym
import numpy as np

"""
Test implementation of a DQL-'agent' that attempts to determine best action to take
for the portfolio.
"""

class PortfolioVenv(gym.Env):

    def __init__(self, df, cash):
        """
        df: dataframes with all OHCLV column of all assets. Should have naming convention
        'Ticker-Open', etc.
        """
        self.df = df
        self.num_assets = (len(df.loc[df.index[0]]) - 1) // 5
        self.init_cash = cash

