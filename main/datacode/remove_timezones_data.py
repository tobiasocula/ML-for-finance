import pandas as pd
import os
import sys

data_dir = '../data/stocks_5m'

for filename in os.listdir(data_dir):
    df = pd.read_csv(os.path.join(data_dir, filename))

    # Convert date column to datetime, parse timezone correctly as UTC
    dt_with_tz = pd.to_datetime(df['Datetime'], utc=True)

    # Remove timezone info to get naive datetime in UTC
    dt_utc_naive = dt_with_tz.dt.tz_localize(None)

    df['Date'] = dt_utc_naive
    df.drop(['Datetime'], axis=1).to_csv(os.path.join(data_dir, filename))
