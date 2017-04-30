import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

from const import SEED
np.random.seed(SEED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--q1-train-output-file', required=True, dest='q1_train_output_file')
    parser.add_argument('--q1-test-output-file', required=True, dest='q1_test_output_file')
    parser.add_argument('--q2-train-output-file', required=True, dest='q2_train_output_file')
    parser.add_argument('--q2-test-output-file', required=True, dest='q2_test_output_file')
    parser.add_argument('--question-col', required=False, dest='question_col', default=None)
    args = parser.parse_args()

    ft = ['question1','question2','question1_porter','question2_porter']
    train = pd.read_csv(args.train_file).astype(str)[ft]
    test = pd.read_csv(args.test_file).astype(str)[ft]

    len_train = train.shape[0]

    data_all = pd.concat([train,test])

    max_features = None
    ngram_range = (1,2)
    min_df = 3
    print('Generate ' + args.question_col + ' tfidf')
    feats = ['question1','question2']
    
    if args.question_col:
        feats = [ '_'.join([x, args.question_col]) for x in feats]
    
    vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

    corpus = []
    for f in feats:
        data_all[f] = data_all[f].astype(str)
        corpus+=data_all[f].values.tolist()

    vect_orig.fit(corpus)

    for f in feats:
        tfidfs = vect_orig.transform(data_all[f].values.tolist())
    
        train_tfidf = tfidfs[:train.shape[0]]
        test_tfidf = tfidfs[train.shape[0]:]

        if 'question1' in f:
            pd.to_pickle(train_tfidf, args.q1_train_output_file)
            pd.to_pickle(test_tfidf, args.q1_test_output_file)
        else:
            pd.to_pickle(train_tfidf, args.q2_train_output_file)
            pd.to_pickle(test_tfidf, args.q2_test_output_file)

