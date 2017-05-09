from itertools import combinations
from scipy.stats import skew, boxcox
import argparse
import logging
import numpy as np
import os
import pandas as pd
import time
from dateutil.relativedelta import relativedelta
from datetime import datetime, date, timedelta
from dateutil.parser import parse
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.model_selection import StratifiedKFold
from nltk.corpus import stopwords
from kaggler.data_io import load_data, save_data
from collections import Counter
from fuzzywuzzy import fuzz


TARGET = 'is_duplicate'


def generate_h1(train_file, test_file,
                train_feature_file, test_feature_file, feature_map_file):

    df_train = pd.read_csv(train_file)
    df_test  = pd.read_csv(test_file)

    print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

    print("Features processing, be patient...")

    df = pd.concat([df_train, df_test])

    x = pd.DataFrame()
    
    x['fuzz_qratio'] = df.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)

    x['fuzz_WRatio'] =df.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)

    x['fuzz_partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])),
                                            axis=1)

    x['fuzz_partial_token_set_ratio'] = df.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)

    x['fuzz_partial_token_sort_ratio'] = df.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)

    x['fuzz_token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])),
                                              axis=1)

    x['fuzz_token_sort_ratio'] = df.apply(
        lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
  
    feature_names = list(x.columns.values)
    print("Features: {}".format(feature_names))

    x.fillna(0, inplace=True)

    x_train = x[:df_train.shape[0]]
    x_test  = x[df_train.shape[0]:]
    y_train = df_train[TARGET].values

    logging.info('saving features')
    save_data(x_train, y_train, train_feature_file)
    save_data(x_test, None, test_feature_file)

    with open(feature_map_file, 'w') as f:
        for i, col in enumerate(x.columns):
            f.write('{}\t{}\tq\n'.format(i, col))


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-feature-file', required=True, dest='trn_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='tst_feature_file')
    parser.add_argument('--feature-map-file', required=True, dest='feature_map_file')

    args = parser.parse_args()

    start = time.time()
    generate_h1(args.train_file,
                args.test_file,
                args.trn_feature_file,
                args.tst_feature_file,
                args.feature_map_file)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))