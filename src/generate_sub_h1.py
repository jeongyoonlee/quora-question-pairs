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

"""https://www.kaggle.com/woters/quora-question-pairs/xgb-starter-12357"""

TARGET = 'is_duplicate'


def generate_h1(train_file, test_file,
                train_feature_file, test_feature_file, feature_map_file):

    df_train = pd.read_csv(train_file)
    df_test  = pd.read_csv(test_file)

    print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))

    print("Features processing, be patient...")

    # If a word appears only once, we ignore it completely (likely a typo)
    # Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
    def get_weight(count, eps=10000, min_count=2):
        return 0 if count < min_count else 1 / (count + eps)

    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}

    stops = set(stopwords.words("english"))

    def word_shares(row):
        q1 = set(str(row['question1']).lower().split())
        q1words = q1.difference(stops)
        if len(q1words) == 0:
            return '0:0:0:0:0'

        q2 = set(str(row['question2']).lower().split())
        q2words = q2.difference(stops)
        
        q1stops = q1.intersection(stops)
        q2stops = q2.intersection(stops)

        shared_words = q1words.intersection(q2words)
        shared_weights = [weights.get(w, 0) for w in shared_words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        sum_total_weighs = np.sum(total_weights)

        R1 = np.sum(shared_weights) / sum_total_weighs if sum_total_weighs!=0 else 0 #tfidf share
        R2 = len(shared_words) / (len(q1words) + len(q2words)) if len(q1words) + len(q2words) > 0 else 0 #count share
        R31 = len(q1stops) / len(q1words) if len(q1words) > 0 else 0 #stops in q1
        R32 = len(q2stops) / len(q2words) if len(q2words) > 0 else 0 #stops in q2
        return '{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32)

    df = pd.concat([df_train, df_test])
    df['word_shares'] = df.apply(word_shares, axis=1, raw=True)

    x = pd.DataFrame()
    x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
    x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
    x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

    x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
    x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
    x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']

    x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
    x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
    x['diff_len'] = x['len_q1'] - x['len_q2']

    x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
    x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
    x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

    x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
    x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
    x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

    x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
    x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
    x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']
   
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