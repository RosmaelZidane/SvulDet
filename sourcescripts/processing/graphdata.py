
import pickle as pkl
import pandas as pd
import numpy as np
import os
import sys

        
from graphdataprocessing import full_run_joern 
from dataprocessing import dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uutils.__utils__ as utls

NUM_JOBS = 1 # put 1 to use the full data
JOB_ARRAY_NUMBER = 0 

# Read Data
df = dataset()
# comment this part latter
# vul = df[df['vul'] == 1].sample(300, random_state= 0) # this was to check whether they are funnction with enough change so that we can keep certain processing defined by svuldet
# nonvul = df[df['vul'] == 0].sample(len(vul), random_state= 0)
# df = pd.concat([vul, nonvul])



df = df.iloc[::-1].reset_index(drop=True)
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """Parallelise svdj functions.

    Example:
    df = svdd.dataset()
    row = df.iloc[180189]  # PAPER EXAMPLE
    row = df.iloc[177860]  # EDGE CASE 1
    preprocess(row)
    """
    savedir_before = utls.get_dir(utls.processed_dir() / row["dataset"] / "before")
    savedir_after = utls.get_dir(utls.processed_dir() / row["dataset"] / "after")

    # Write C Files
    fpath1 = savedir_before / f"{row['id']}.java"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.java"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])
    # Run Joern on "before" code
    if not os.path.exists(f"{fpath1}.edges.json"):
        full_run_joern(fpath1, verbose=3)

    # Run Joern on "after" code
    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        full_run_joern(fpath2, verbose=3)
    
        

if __name__ == "__main__":
    utls.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)