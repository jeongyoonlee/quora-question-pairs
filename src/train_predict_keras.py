#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import os
import pandas as pd
import time

from kaggler.data_io import load_data
SEED = 2017


np.random.seed(SEED) # for reproducibility


def get_model(nb_classes, dims, hiddens=2, neurons=512, dropout=0.5):
    model = Sequential()
    model.add(Dense(neurons, input_dim=dims, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    for i in range(hiddens):
        model.add(Dense(neurons, init='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

    model.add(Dense(nb_classes, init='glorot_uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")

    return model


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  cv_id_file, n_est=100, hiddens=2, neurons=512, dropout=0.5, batch=16,
                  n_stop=2, retrain=True, n_fold=5):

    feature_name = os.path.basename(train_file).split('.')[0]
    model_name = 'keras_{}_{}_{}_{}_{}_{}_{}'.format(
        n_est, hiddens, neurons, dropout, batch, n_stop, feature_name
    )

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='{}.log'.format(model_name))

    logging.info('Loading training and test data...')
    X, y = load_data(train_file, dense=True)
    Y = np_utils.to_categorical(y)
    X_tst, _ = load_data(test_file, dense=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tst = scaler.transform(X_tst)

    nb_classes = Y.shape[1]
    dims = X.shape[1]
    logging.info('{} classes, {} dims'.format(nb_classes, dims))

    logging.info('Loading CV Ids')
    cv_id = np.loadtxt(cv_id_file)

    P_val = np.zeros((Y.shape[0], ))
    P_tst = np.zeros((X_tst.shape[0], ))
    for i in range(1, n_fold + 1):
        i_trn = np.where(cv_id != i)[0]
        i_val = np.where(cv_id == i)[0]
        logging.info('Training model #{}'.format(i))
        clf = get_model(nb_classes, dims, hiddens, neurons, dropout)
        if i == 1:
            early_stopping = EarlyStopping(monitor='val_loss', patience=n_stop)
            h = clf.fit(X[i_trn],
                        Y[i_trn],
                        validation_data=(X[i_val], Y[i_val]),
                        nb_epoch=n_est,
                        batch_size=batch,
                        callbacks=[early_stopping])

            val_losses = h.history['val_loss']
            n_best = val_losses.index(min(val_losses)) + 1
            logging.info('best epoch={}'.format(n_best))
        else:
            clf.fit(X[i_trn],
                    Y[i_trn],
                    validation_data=(X[i_val], Y[i_val]),
                    nb_epoch=n_best,
                    batch_size=batch)

        P_val[i_val] = clf.predict_proba(X[i_val])[:, 1]
        logging.info('CV #{} Log Loss: {:.6f}'.format(i, log_loss(Y[i_val], P_val[i_val])))

        if not retrain:
            P_tst += clf.predict_proba(X_tst)[:, 1] / n_fold

    logging.info('CV Log Loss: {:.6f}'.format(log_loss(y, P_val)))
    np.savetxt(predict_valid_file, P_val, fmt='%.6f', delimiter=',')

    if retrain:
        logging.info('Retraining with 100% training data')
        clf = get_model(nb_classes, dims, hiddens, neurons, dropout)
        clf.fit(X, Y, nb_epoch=n_best, batch_size=batch)
        P_tst = clf.predict_proba(X_tst)[:, 1]

    logging.info('Saving normalized test predictions...')
    np.savetxt(predict_test_file, P_tst, fmt='%.6f', delimiter=',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', default=10, type=int, dest='n_est')
    parser.add_argument('--batch-size', default=64, type=int,
                        dest='batch_size')
    parser.add_argument('--hiddens', default=2, type=int)
    parser.add_argument('--neurons', default=512, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--early-stopping', default=2, type=int, dest='n_stop')
    parser.add_argument('--retrain', default=False, action='store_true')
    parser.add_argument('--cv-id', required=True, dest='cv_id_file')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  cv_id_file=args.cv_id_file,
                  n_est=args.n_est,
                  neurons=args.neurons,
                  dropout=args.dropout,
                  batch=args.batch_size,
                  hiddens=args.hiddens,
                  n_stop=args.n_stop,
                  retrain=args.retrain)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))

