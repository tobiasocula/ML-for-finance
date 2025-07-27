"""
removes csv files where volume isn't a column when it should be there
"""

import pandas as pd
from pathlib import Path
import os

root = Path.cwd()
dir = root/'main'/'data'/'stocks_1d'

for k in dir.iterdir():
    df = pd.read_csv(k)
    if 'Volume' not in df.columns:
        os.remove(k)        