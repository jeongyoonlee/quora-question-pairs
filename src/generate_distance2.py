import numpy as np
import pandas as pd
import sklearn.metrics.pairwise
from kaggler.data_io import load_data, save_data
import argparse


def calculate_distance(vec_files):
    print vec_files
    q1_vec, _ = load_data(vec_files[0])
    q2_vec, _ = load_data(vec_files[1])

    distances = []

    for d in sklearn.metrics.pairwise.PAIRED_DISTANCES.keys(): #['euclidean', 'cosine', 'l2', 'l1', 'cityblock', 'manhattan']
        distances.append(sklearn.metrics.pairwise.paired_distances(q1_vec, q2_vec, metric=d))

    return np.transpose(np.vstack(distances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-vec-files', required=True, dest='train_files')
    parser.add_argument('--test-vec-files', required=True, dest='test_files')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    args = parser.parse_args()

    train_dis = calculate_distance(args.train_files.split(' '))
    test_dis = calculate_distance(args.test_files.split(' '))

    save_data(train_dis, None, args.train_output_file)
    save_data(test_dis, None, args.test_output_file)

