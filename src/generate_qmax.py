import argparse
import pandas as pd
from kaggler.data_io import save_data
import networkx as nx
# https://www.kaggle.com/ashhafez/temporal-pattern-in-train-response-rates
# https://www.kaggle.com/tezdhar/temporal-pattern-verified-with-quora-dates

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_file).astype(str)
    df_test = pd.read_csv(args.test_file).astype(str)

    mapping = {}

    df_train["qmax"] = df_train.apply( lambda row: max( mapping.setdefault(row["question1"], len(mapping)), 
                                                        mapping.setdefault(row["question2"], len(mapping))), axis=1 )
    df_test["qmax"] = df_test.apply( lambda row: max( mapping.setdefault(row["question1"], len(mapping)), 
                                                      mapping.setdefault(row["question2"], len(mapping))), axis=1 )

    mapping_porter = {}
    df_train["qmax_porter"] = df_train.apply( lambda row: max( mapping_porter.setdefault(row["question1_porter"], len(mapping_porter)), 
                                                               mapping_porter.setdefault(row["question2_porter"], len(mapping_porter))), axis=1 )
    df_test["qmax_porter"] = df_test.apply( lambda row: max( mapping_porter.setdefault(row["question1_porter"], len(mapping_porter)), 
                                                             mapping_porter.setdefault(row["question2_porter"], len(mapping_porter))), axis=1 )

    features = ["qmax", "qmax_porter"]

    save_data(df_train[features].values, None, args.train_output_file)
    save_data(df_test[features].values, None, args.test_output_file)
