#!/usr/bin/env python
from __future__ import division

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from const import SEED

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        dest='input_file')
    parser.add_argument('--cv', '-c', required=True,
                        dest='cv_file')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    y = df["is_duplicate"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_id = np.zeros_like(y, dtype=int)
    for i, (i_trn, i_val) in enumerate(cv.split(trn, y), 1):
        cv_id[i_val] = i

    np.savetxt(cv_file, cv_id, fmt='%d')
