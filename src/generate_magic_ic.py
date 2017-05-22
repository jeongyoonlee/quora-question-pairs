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
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    df = pd.concat([train, test])

    g = nx.Graph()
    g.add_nodes_from(df.question1)
    g.add_nodes_from(df.question2)
    edges = list(df[['question1', 'question2']].to_records(index=False))
    g.add_edges_from(edges)


    def get_intersection_count(row):
        return(len(set(g.neighbors(row.question1)).intersection(set(g.neighbors(row.question2)))))

    train_ic = pd.DataFrame()
    test_ic = pd.DataFrame()

    train['intersection_count'] = train.apply(lambda row: get_intersection_count(row), axis=1)
    test['intersection_count'] = test.apply(lambda row: get_intersection_count(row), axis=1)

    save_data(train[['intersection_count']].values, None, args.train_output_file)
    save_data(test[['intersection_count']].values, None, args.test_output_file)
