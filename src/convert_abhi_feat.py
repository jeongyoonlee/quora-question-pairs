from __future__ import division

import argparse
import pandas as pd
import numpy as np
import os
import sklearn.metrics as sm
from kaggler.data_io import load_data, save_data

FEATURES = ['len_q1','len_q2','diff_len','len_char_q1','len_char_q2','len_word_q1','len_word_q2','common_words', \
             'fuzz_qratio','fuzz_WRatio','fuzz_partial_ratio','fuzz_partial_token_set_ratio','fuzz_partial_token_sort_ratio','fuzz_token_set_ratio','fuzz_token_sort_ratio', \
             'wmd','norm_wmd',\
             'cosine_distance','cityblock_distance','jaccard_distance','canberra_distance','euclidean_distance','minkowski_distance','braycurtis_distance',\
             'skew_q1vec','skew_q2vec','kur_q1vec','kur_q2vec']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True,
                        dest='input_file')
    parser.add_argument('--output', '-o', required=True,
                        dest='output_file')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)

    save_data(df[FEATURES].values, None, args.output_file)
