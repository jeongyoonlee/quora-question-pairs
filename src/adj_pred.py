from __future__ import division

import argparse
import pandas as pd
import numpy as np
import os
import sklearn.metrics as sm

# https://github.com/0celot/mlworkshop39_042017/blob/master/3_masterclass/ipy/feature_extraction.ipynb

gamma_0 = 1.30905513329
gamma_1 = 0.472008228977

def link_function(x):
    return gamma_1*x/(gamma_1*x + gamma_0*(1 - x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        dest='input_file')
    parser.add_argument('--output', '-o', required=True,
                        dest='output_file')
    args = parser.parse_args()

    p = np.loadtxt(args.input_file, delimiter=',')

    p_fixed = link_function(p)

    np.savetxt(args.output_file, p_fixed, fmt='%.10f')



