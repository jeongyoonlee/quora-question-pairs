import argparse
import pandas as pd
from kaggler.data_io import save_data
import networkx as nx
#https://www.kaggle.com/justfor/edges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    parser.add_argument('--question-col', required=False, dest='question_col', default=None)
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    df = pd.concat([train, test])

    q1_col = 'question1'
    q2_col = 'question2'

    if args.question_col:
        q1_col += '_' + args.question_col
        q2_col += '_' + args.question_col

    g = nx.Graph()
    g.add_nodes_from(df[q1_col])
    g.add_nodes_from(df[q2_col])
    edges = list(df[[q1_col, q2_col]].to_records(index=False))
    g.add_edges_from(edges)


    # def get_intersection_count(row):
    #     return(len(set(g.neighbors(row[q1_col])).intersection(set(g.neighbors(row[q2_col])))))

    def get_union(row):
        return len(set(g.neighbors(row[q1_col])).union(set(g.neighbors(row[q2_col]))))

    def get_d1(row):
        return len(set(g.neighbors(row[q1_col])).difference(set(g.neighbors(row[q2_col]))))

    def get_d2(row):
        return len(set(g.neighbors(row[q2_col])).difference(set(g.neighbors(row[q1_col]))))

    def get_symmetric_diff(row):
        return len(set(g.neighbors(row[q2_col])).symmetric_difference(set(g.neighbors(row[q1_col]))))
    
    print("union")
    train['union_count'] = train.apply(lambda row: get_union(row), axis=1)
    test['union_count'] = test.apply(lambda row: get_union(row), axis=1)

    print("d1_count")
    train['d1_count'] = train.apply(lambda row: get_d1(row), axis=1)
    test['d1_count'] = test.apply(lambda row: get_d1(row), axis=1)

    print("d2_count")
    train['d2_count'] = train.apply(lambda row: get_d2(row), axis=1)
    test['d2_count'] = test.apply(lambda row: get_d2(row), axis=1)

    print("sy_diff_count")
    train['sy_diff_count'] = train.apply(lambda row: get_symmetric_diff(row), axis=1)
    test['sy_diff_count'] = test.apply(lambda row: get_symmetric_diff(row), axis=1)

    features = ['union_count', 'd1_count', 'd2_count', 'sy_diff_count']

    save_data(train[features].values, None, args.train_output_file)
    save_data(test[features].values, None, args.test_output_file)
