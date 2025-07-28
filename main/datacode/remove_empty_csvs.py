from pathlib import Path
import os
import pandas as pd

root = Path.cwd()
datadir = root/'main'/'data'

workdir = datadir/'stocks_1d'

for file in workdir.iterdir():
    df = pd.read_csv(str(workdir/file))
    if df.empty:
        os.remove(str(workdir/file))
        print('removed', file.name)