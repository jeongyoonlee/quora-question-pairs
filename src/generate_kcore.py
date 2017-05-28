import argparse
import pandas as pd
from kaggler.data_io import save_data
import networkx as nx
# https://www.kaggle.com/c/quora-question-pairs/discussion/33371

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    parser.add_argument('--question-col', required=False, dest='question_col', default=None)

    args = parser.parse_args()

    print('Generate ' + str(args.question_col) + ' kcores')

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    df = pd.concat([train, test])

    g = nx.Graph()

    q1_col = 'question1'
    q2_col = 'question2'

    if args.question_col:
        q1_col += '_' + args.question_col
        q2_col += '_' + args.question_col

    g.add_nodes_from(df[q1_col])
    g.add_nodes_from(df[q2_col])
    edges = list(df[[q1_col, q2_col]].to_records(index=False))
    g.add_edges_from(edges)

    g.remove_edges_from(g.selfloop_edges())

    df_output = pd.DataFrame(data=g.nodes(), columns=["question"])
    print("df_output.shape:", df_output.shape)

    NB_CORES = 20

    for k in range(2, NB_CORES + 1):
        fieldname = "kcore{}".format(k)
        print("fieldname = ", fieldname)
        ck = nx.k_core(g, k=k).nodes()
        print("len(ck) = ", len(ck))
        df_output[fieldname] = 0
        df_output.ix[df_output.question.isin(ck), fieldname] = k

    df_output.set_index("question", inplace=True)
    df_output['max_kcore'] = df_output.apply(lambda row: max(row), axis=1)

    cores_dict = df_output.to_dict()["max_kcore"]
    
    def gen_qid1_max_kcore(row):
        return cores_dict[row[q1_col]]

    def gen_qid2_max_kcore(row):
        return cores_dict[row[q2_col]]

    def gen_max_kcore(row):
       return max(row["qid1_max_kcore"], row["qid2_max_kcore"])

    train["qid1_max_kcore"] = train.apply(gen_qid1_max_kcore, axis=1)
    test["qid1_max_kcore"] = test.apply(gen_qid1_max_kcore, axis=1)
    
    train["qid2_max_kcore"] = train.apply(gen_qid2_max_kcore, axis=1)
    test["qid2_max_kcore"] = test.apply(gen_qid2_max_kcore, axis=1)
    
    train["max_kcore"] = train.apply(gen_max_kcore, axis=1)
    test["max_kcore"] = test.apply(gen_max_kcore, axis=1)

    features = ["qid1_max_kcore", "qid2_max_kcore", "max_kcore"]

    save_data(train[features].values, None, args.train_output_file)
    save_data(test[features].values, None, args.test_output_file)
