"""
function for finding valid date range for n given dataframes such that they all lign up
assuming df.index is DatetimeIndex
"""

import pandas as pd
import os, sys
from .find_relevant_csv import find_relevant_csv

def correct_csvs(tickers, data_dirs):
    dfs = []
    for t, d in zip(tickers, data_dirs):
        df = find_relevant_csv(d, t)
        dfs.append(df)
    
    # Find intersected index (common datetimes) across all dataframes
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    
    res = []
    for df in dfs:
        # Filter dataframe to only include rows with common index values
        filtered_df = df.loc[common_index]
        res.append(filtered_df)
        print('Filtered DataFrame:')
        print(filtered_df)
        print()
    return res