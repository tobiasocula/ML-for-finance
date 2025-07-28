from pathlib import Path
import os

root = Path.cwd()
datadir = root/'main'/'data'

workdir = datadir/'stocks_1d'

for file in workdir.iterdir():
    if 'nat' in file.name.lower():
        os.remove(str(workdir/file))