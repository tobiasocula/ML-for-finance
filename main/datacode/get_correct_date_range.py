"""
function for finding valid date range for n given dataframes such that they all lign up
assuming df.index is DatetimeIndex
"""

import pandas as pd
import os, sys
from .find_relevant_csv import find_relevant_csv

def intersecting_timestamps(tickers, data_dirs):
    dfs = []
    mindate, maxdate = None, None
    for t, d in zip(tickers, data_dirs):
        df = find_relevant_csv(d, t)
        df.index = pd.to_datetime(df.index)
        mindate = max(df.index.min(), mindate) if mindate is not None else df.index.min()
        maxdate = min(df.index.max(), maxdate) if maxdate is not None else df.index.max()
        print(f"Ticker: {t}, Min date: {df.index.min()}, Max date: {df.index.max()}")
        print('current min and max:', mindate, maxdate)
        dfs.append(df)
    
    # Find intersected index (common datetimes) across all dataframes
    common_index = dfs[0].index
    print('common index:', common_index)
    for i, df in enumerate(dfs[1:], 1):
        common_index = common_index.intersection(df.index)
        print('current index type:', type(df.index[0]))
        if len(common_index) == 0:
            
            raise ValueError(f"""No overlapping dates between the selected tickers: min:
                             {df.index.min()}, max: {df.index.max()}. Failed on ticker {tickers[i]}""")

        print('minmax indeces:', df.index.min(), df.index.max())
        print()
        """
        Index(['2014-09-17', '2014-09-18', '2014-09-19', '2014-09-22', '2014-09-23',
       '2014-09-24', '2014-09-25', '2014-09-26', '2014-09-29', '2014-09-30',
       ...
       '2025-07-14', '2025-07-15', '2025-07-16', '2025-07-17', '2025-07-18',
       '2025-07-21', '2025-07-22', '2025-07-23', '2025-07-24', '2025-07-25'],
       """
    
    res = []
    for df in dfs:
        # Filter dataframe to only include rows with common index values
        filtered_df = df.loc[common_index]
        res.append(filtered_df)
    return res

def contains_range_timestamps(tickers, data_dirs, min_date, max_date):
    """
    Function for fetching & filtering csv files. Filters based on datetimeindex being between
    min_date and max_date. Assumes tickers are unique in their data_dirs.
    """
    dfs = []
    for t, d in zip(tickers, data_dirs):
        df = find_relevant_csv(d, t)
        df.index = pd.to_datetime(df.index)
        if min_date <= df.index.min() and df.index.max() <= max_date:
            dfs.append(df)
    return dfs