from pathlib import Path
import os
import pandas as pd

root = Path.cwd()
datadir = root/'main'/'data'

workdir = datadir/'stocks_5m'

for file in workdir.iterdir():
    df = pd.read_csv(str(workdir/file))
    if df.empty:
        os.remove(str(workdir/file))
        print('removed', file.name)