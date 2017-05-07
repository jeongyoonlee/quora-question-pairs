import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import distance
from nltk.corpus import stopwords
import nltk
import argparse
from kaggler.data_io import load_data, save_data
from kaggler import feature_selection
import logging
from scipy import sparse as ssp
import time
from IPython import embed

TARGET = 'is_duplicate'


def merge_sub_features(train_file, test_file, 
                       train_sub_features, test_sub_features,
                       train_feature_file, test_feature_file,
                       lowest):

    trn_subfeat = []
    tst_subfeat = []

    for f_trn, f_tst in zip([ x for x in train_sub_features.split(' ') if x], 
                            [ x for x in test_sub_features.split(' ') if x]):
        logging.info('Reading trn {0} tst {1}'.format(f_trn, f_tst))
        
        X_sub_trn, _ = load_data(f_trn)
        X_sub_tst, _ = load_data(f_tst)

        if not ssp.issparse(X_sub_trn):
            X_sub_trn = ssp.csr_matrix(X_sub_trn)
            X_sub_tst = ssp.csr_matrix(X_sub_tst)

        trn_subfeat.append(X_sub_trn)
        tst_subfeat.append(X_sub_tst)

        logging.info('Size trn {0} tst {1}'.format(X_sub_trn.shape, X_sub_tst.shape))

    df_train = pd.read_csv(train_file)
    y_train = df_train[TARGET].values

    logging.info('Merge sub features')
    X_trn = ssp.hstack(trn_subfeat).tocsr()
    X_tst = ssp.hstack(tst_subfeat).tocsr()
    logging.info('Size trn {0} tst {1}'.format(X_trn.shape, X_tst.shape))

    drop = feature_selection.DropInactive(lowest)

    drop.fit(X_trn)
    X_trn = drop.transform(X_trn)
    X_tst = drop.transform(X_tst)

    logging.info('Size trn {0} tst {1}'.format(X_trn.shape, X_tst.shape))
    
    logging.info('saving features')
    save_data(X_trn, y_train, train_feature_file)
    save_data(X_tst, None, test_feature_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-sub-features', required=True, dest='train_sub_features')
    parser.add_argument('--test-sub-features', required=True, dest='test_sub_features')
    parser.add_argument('--train-feature-file', required=True, dest='train_output_file')
    parser.add_argument('--test-feature-file', required=True, dest='test_output_file')
    parser.add_argument('--lowest', required=True, dest='lowest', type=float)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG,
                        filename='merge.log')

    start = time.time()
    merge_sub_features(args.train_file,
                        args.test_file,
                        args.train_sub_features,
                        args.test_sub_features,
                        args.train_output_file,
                        args.test_output_file,
                        args.lowest)
    logging.info('finished ({:.2f} sec elasped)'.format(time.time() - start))

