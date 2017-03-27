#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data

import xgboost as xgb
from const import SEED


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  cv_id_file, n_est=100, depth=4, lrate=.1, subcol=.5, subrow=.5, sublev=1,
                  weight=1, n_stop=100, retrain=True, n_fold=5):

    feature_name = os.path.basename(train_file).split('.')[0]
    model_name = 'xg_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
        n_est, depth, lrate, subcol, subrow, sublev, weight, n_stop, feature_name
    )

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    # set xgb parameters
    params = {'objective': "binary:logistic",
              'max_depth': depth,
              'eta': lrate,
              'subsample': subrow,
              'colsample_bytree': subcol,
              'colsample_bylevel': sublev,
              'min_child_weight': weight,
              'silent': 1,
              'nthread': 10,
              'seed': SEED,
              'eval_metric': 'logloss'}

    logging.info('Loading training and test data...')
    X, y = load_data(train_file)
    X_tst, _ = load_data(test_file)
    xgtst = xgb.DMatrix(X_tst)

    logging.info('Loading CV Ids')
    cv_id = np.loadtxt(cv_id_file)

    P_val = np.zeros(X.shape[0])
    P_tst = np.zeros(X_tst.shape[0])
    for i in range(1, n_fold + 1):
        i_trn = np.where(cv_id != i)[0]
        i_val = np.where(cv_id == i)[0]
        logging.debug('train: {}'.format(X[i_trn].shape))
        logging.debug('valid: {}'.format(X[i_val].shape))
        xgtrn = xgb.DMatrix(X[i_trn], label=y[i_trn])
        xgval = xgb.DMatrix(X[i_val], label=y[i_val])

        logging.info('Training model #{}'.format(i))
        watchlist = [(xgtrn, 'train'), (xgval, 'val')]

        if i == 1:
            logging.info('Training with early stopping')
            clf = xgb.train(params, xgtrn, n_est, watchlist,
                            early_stopping_rounds=n_stop)
            n_best = clf.best_iteration
            logging.info('best iteration={}'.format(n_best))
        else:
            clf = xgb.train(params, xgtrn, n_best, watchlist)

        P_val[i_val] = clf.predict(xgval, ntree_limit=n_best)

        if not retrain:
            P_tst += clf.predict(xgtst, ntree_limit=n_best) / n_fold

    logging.info('Saving validation predictions...')
    np.savetxt(predict_valid_file, P_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        xgtrn = xgb.DMatrix(X, label=y)
        watchlist = [(xgtrn, 'train')]
        clf = xgb.train(params, xgtrn, n_best, watchlist)
        P_tst = clf.predict(xgtst, ntree_limit=n_best)

    logging.info('Saving test predictions...')
    np.savetxt(predict_test_file, P_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', type=int, dest='n_est')
    parser.add_argument('--depth', type=int)
    parser.add_argument('--lrate', type=float)
    parser.add_argument('--subcol', type=float, default=1)
    parser.add_argument('--subrow', type=float, default=.5)
    parser.add_argument('--sublev', type=float, default=1.)
    parser.add_argument('--weight', type=int, default=1)
    parser.add_argument('--early-stop', type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--cv-id', required=True, dest='cv_id_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  depth=args.depth,
                  lrate=args.lrate,
                  subcol=args.subcol,
                  subrow=args.subrow,
                  sublev=args.sublev,
                  weight=args.weight,
                  n_stop=args.n_stop,
                  retrain=args.retrain,
                  cv_id_file=args.cv_id_file)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
