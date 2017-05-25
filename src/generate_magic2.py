import argparse
import pandas as pd
from kaggler.data_io import save_data
import numpy as np

def generate_frequency_features(train, test, question1_col, question2_col):
    df1 = train[[question1_col]].copy()
    df2 = train[[question2_col]].copy()

    df1_test = test[[question1_col]].copy()
    df2_test = test[[question2_col]].copy()

    df2.rename(columns={question2_col: question1_col}, inplace=True)
    df2_test.rename(columns={question2_col: question1_col}, inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    # train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
    train_questions.drop_duplicates(subset=[question1_col], inplace=True)

    train_questions.reset_index(inplace=True, drop=True)
    questions_dict = pd.Series(train_questions.index.values, index=train_questions[question1_col].values).to_dict()
    train_cp = train.copy()
    test_cp = test.copy()
    train_cp.drop(['qid1', 'qid2'], axis=1, inplace=True)

    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id': 'id'}, inplace=True)
    comb = pd.concat([train_cp, test_cp])

    comb[question1_col+'_hash'] = comb[question1_col].map(questions_dict)
    comb[question2_col+'_hash'] = comb[question2_col].map(questions_dict)

    q1_vc = comb[question1_col+'_hash'].value_counts().to_dict()
    q2_vc = comb[question2_col+'_hash'].value_counts().to_dict()

    features = []
    # map to frequency space
    print "freq"
    comb[question1_col+'_freq'] = comb[question1_col+'_hash'].map(lambda x: q1_vc.get(x, 0) + q2_vc.get(x, 0))
    comb[question2_col+'_freq'] = comb[question2_col+'_hash'].map(lambda x: q1_vc.get(x, 0) + q2_vc.get(x, 0))
    features.append(question1_col+'_freq')
    features.append(question2_col+'_freq')
    
    print "min max mean ..."
    comb['freq_min'] = comb[[question1_col+'_freq', question2_col+'_freq']].min(axis=1)
    comb['freq_max'] = comb[[question1_col+'_freq', question2_col+'_freq']].max(axis=1)
    comb['freq_meam'] = comb[[question1_col+'_freq', question2_col+'_freq']].mean(axis=1)

    print "diff"
    comb['freq_diff1'] = comb[question1_col+'_freq'] - comb[question2_col+'_freq']
    comb['freq_diff2'] = comb[question2_col+'_freq'] - comb[question1_col+'_freq']
    
    features.append('freq_min')
    features.append('freq_max')
    features.append('freq_meam')
    features.append('freq_diff1')
    features.append('freq_diff2')

    train_comb = comb[comb['is_duplicate'] >= 0][features]
    test_comb = comb[comb['is_duplicate'] < 0][features]

    return train_comb, test_comb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    train_feats, test_feats = generate_frequency_features(train, test, 'question1', 'question2')
    train_feats_porter, test_feats_porter = generate_frequency_features(train, test, 'question1_porter', 'question2_porter')

    train_magic = pd.concat([train_feats, train_feats_porter], axis=1)
    test_magic = pd.concat([test_feats, test_feats_porter], axis=1)

    save_data(train_magic.values, None, args.train_output_file)
    save_data(test_magic.values, None, args.test_output_file)
