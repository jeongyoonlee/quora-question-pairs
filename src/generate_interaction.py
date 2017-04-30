import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse

from const import SEED
np.random.seed(SEED)

def calc_set_intersection(text_a, text_b):
    a = set(text_a.split())
    b = set(text_b.split())
    return len(a.intersection(b)) *1.0 / len(a)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    feats = []

    print('Generate intersection')
    train['question_intersection'] = train.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
    test['question_intersection'] = test.astype(str).apply(lambda x:calc_set_intersection(x['question1'],x['question2']),axis=1)
    feats.append('question_intersection')

    print('Generate porter intersection')
    train['question_porter_intersection']train_porter_interaction = train.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
    test['question_porter_intersection'] = test.astype(str).apply(lambda x:calc_set_intersection(x['question1_porter'],x['question2_porter']),axis=1)
    feats.append('question_porter_intersection')

    pd.to_pickle(train[feats].values, args.train_output_file)
    pd.to_pickle(test[feats].values, args.test_output_file)
