import argparse
import pandas as pd
import numpy as np
from kaggler.data_io import load_data, save_data
import cPickle as pickle
from scipy import sparse as ssp
import logging

TARGET = 'is_duplicate'

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--input-file1', required=True, dest='input_file1')
    parser.add_argument('--input-file2', required=True, dest='input_file2')
    parser.add_argument('--train-feature-file', required=True, dest='train_feature_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_feature_file')
    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file)
    print len(df_train)
    y_train = df_train[TARGET].values

    X1 = load_pickle(args.input_file1)
    print X1.shape
    X2 = load_pickle(args.input_file2)
    print X2.shape

    X = ssp.hstack([X1, X2]).tocsr()
    print X.shape

    logging.info('saving features')
    save_data(X1[:len(y_train),:], y_train, args.train_feature_file)
    save_data(X1[len(y_train):,:], None, args.test_feature_file)
    