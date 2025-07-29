
from pathlib import Path
from datetime import datetime
import numpy as np
import os

def filter(dirname: Path):

    """Removes csv files in specified dirname that are abundant (have a shorter timespan than
    the same symbol in another csv file)."""
    files = list(dirname.iterdir())
    
    i = 0
    dt_format = "%Y-%m-%d:%H-%M-%S"
    len_files = len(files)
    while i < len_files-1:
        main_filename = files[i].name
        main_splitted = main_filename.split('--')
        mindate = datetime.strptime(main_splitted[1] + ':' + main_splitted[2], dt_format)
        maxdate = datetime.strptime(main_splitted[3] + ':' + main_splitted[4], dt_format)
        j = i + 1
        next_filename = files[i+1].name
        next_splitted = next_filename.split('--')
        
        curr_iter = [
            datetime.strptime(main_splitted[3] + ':' + main_splitted[4], dt_format)
            -
            datetime.strptime(main_splitted[1] + ':' + main_splitted[2], dt_format)
        ] # stores date ranges in order
        curr_iter_names = [files[i].name]

        while main_splitted[0] == next_splitted[0]:
            mindate = min(mindate, datetime.strptime(next_splitted[1] + ':' + next_splitted[2], dt_format))
            maxdate = min(maxdate, datetime.strptime(next_splitted[3] + ':' + next_splitted[4], dt_format))
            curr_iter.append(
                datetime.strptime(next_splitted[3] + ':' + next_splitted[4], dt_format)
                -
                datetime.strptime(next_splitted[1] + ':' + next_splitted[2], dt_format)
            )
            curr_iter_names.append(
                files[j].name
            )
                
            next_filename = files[j].name
            next_splitted = next_filename.split('--')
            j += 1

        # from curr_iter, store files to delete
        max_len_idx = np.argmax(curr_iter)
        for k, tbd in enumerate(curr_iter_names):
            if k != max_len_idx:
                os.remove(str(dirname/Path(tbd)))
        i = j

            

