import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import argparse

from const import SEED
np.random.seed(SEED)
from kaggler.data_io import load_data, save_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--q1-train-output-file', required=True, dest='q1_train_output_file')
    parser.add_argument('--q1-test-output-file', required=True, dest='q1_test_output_file')
    parser.add_argument('--q2-train-output-file', required=True, dest='q2_train_output_file')
    parser.add_argument('--q2-test-output-file', required=True, dest='q2_test_output_file')
    args = parser.parse_args()

    feats = ['question1','question2']
    train = pd.read_csv(args.train_file).astype(str)[feats]
    test = pd.read_csv(args.test_file).astype(str)[feats]

    len_train = train.shape[0]

    data_all = pd.concat([train,test])

    print('Generate bow char')
    
    bow_extractor = CountVectorizer(max_df=0.999, min_df=500, 
                                    max_features=300000, 
                                    analyzer='char', ngram_range=(1,9), 
                                    binary=True, lowercase=True)

    corpus = []
    for f in feats:
        data_all[f] = data_all[f].astype(str)
        corpus+=data_all[f].values.tolist()

    bow_extractor.fit(corpus)

    for f in feats:
        bow = bow_extractor.transform(data_all[f].values.tolist())
    
        train_bow = bow[:train.shape[0]]
        test_bow = bow[train.shape[0]:]

        if 'question1' in f:
            save_data(train_bow, None, args.q1_train_output_file)
            save_data(test_bow, None, args.q1_test_output_file)
        else:
            save_data(train_bow, None, args.q2_train_output_file)
            save_data(test_bow, None, args.q2_test_output_file)

