import nltk
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
import pandas as pd
from kaggler.data_io import save_data
import argparse
from utils.phrase_vector import PhraseVector


def process_pos_tags(tokens):
    tagged_words = {'CC': [], 'NNP': [], 'NN': [], 'VB': [], 'RB': [], 'JJ': [], 'W': []}

    for token in tokens:
        if 'CC' in token[1]:
            tagged_words['CC'].append(token[0])
        elif 'NNP' in token[1]:
            tagged_words['NNP'].append(token[0])
        elif 'NN' in token[1]:
            tagged_words['NN'].append(token[0])
        elif 'VB' in token[1]:
            tagged_words['VB'].append(token[0])
        elif 'RB' in token[1]:
            tagged_words['RB'].append(token[0])
        elif 'JJ' in token[1]:
            tagged_words['JJ'].append(token[0])
        elif 'W' in token[1]:
            tagged_words['W'].append(token[0])

    return tagged_words


def compute_features(tag1, tag2):
    considered_parts = ['CC', 'NNP', 'NN', 'VB', 'RB', 'JJ', 'W']
    res = {}
    for part in considered_parts:
        words1 = set(tag1[part])
        words2 = set(tag2[part])

        res[part + '_q1_len'] = len(words1)
        res[part + '_q2_len'] = len(words2)

        if len(words1) == 0 or len(words2) == 0:
            res[part + '_wordmatch_percentage'] = 0
        else:
            res[part + '_wordmatch_percentage'] = len(words1.intersection(words2)) / float(len(words1.union(words2)))

        p1 = PhraseVector(words1, True)
        p2 = PhraseVector(words2, True)
        res[part + '_word2vec'] = p1.CosineSimilarity(p2.vector)

    return res


def pos_features(row):
    question1 = word_tokenize(row[4].lower())
    question2 = word_tokenize(row[5].lower())

    tag1 = pos_tag(question1)
    ptag1 = process_pos_tags(tag1)
    tag2 = pos_tag(question2)
    ptag2 = process_pos_tags(tag2)

    d = compute_features(ptag1, ptag2)

    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    df_train = train.apply(pos_features, axis=1, raw=True)
    df_train = df_train.apply(pd.Series)
    df_test = test.apply(pos_features, axis=1, raw=True)
    df_test = df_test.apply(pd.Series)  

    save_data(df_train.values, None, args.train_output_file)
    save_data(df_test.values, None, args.test_output_file)