import os
import pandas as pd

def find_relevant_csv(dir, symbol):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # current script folder
    datafolder = os.path.join(os.path.dirname(script_dir), dir)

    for filename in os.listdir(datafolder):
        if symbol in filename:
            return pd.read_csv(os.path.join(datafolder, filename), index_col=0)