import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import distance
from nltk.corpus import stopwords
import nltk
import argparse
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import keras.layers as lyr
from keras.models import Model
from kaggler.data_io import save_data


class LSTMModel:

    def fit(self, X1_train, X2_train, X1_val, X2_val, y_train, y_val):
        self.input1_tensor = lyr.Input(X1_train.shape[1:])
        self.input2_tensor = lyr.Input(X2_train.shape[1:])

        self.words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
        self.seq_embedding_layer = lyr.LSTM(256, activation='tanh')

        self.seq_embedding = lambda tensor: self.seq_embedding_layer(self.words_embedding_layer(tensor))

        self.merge_layer = lyr.multiply([self.seq_embedding(self.input1_tensor), self.seq_embedding(self.input2_tensor)])

        dense1_layer = lyr.Dense(16, activation='sigmoid')(self.merge_layer)
        ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

        self.model = Model([self.input1_tensor, self.input2_tensor], ouput_layer)

        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        print self.model.summary()

        self.model.fit([X1_train, X2_train], y_train,
                  validation_data=([X1_val, X2_val], y_val),
                  batch_size=128, epochs=6, verbose=2)

        self.features_model = Model([self.input1_tensor, self.input2_tensor], self.merge_layer)
        self.features_model.compile(loss='mse', optimizer='adam')

    def extractFeatures(self, X1, X2):
        return self.features_model.predict([X1, X2], batch_size=128)


def create_padded_seqs(texts, max_len=10):
    words_tokenizer = re.compile(counts_vectorizer.token_pattern)
    seqs = texts.apply(lambda s:
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--train-output-file', required=True, dest='train_output_file')
    parser.add_argument('--test-output-file', required=True, dest='test_output_file')
    args = parser.parse_args()

    train = pd.read_csv(args.train_file).astype(str)
    test = pd.read_csv(args.test_file).astype(str)

    train['id'] = train['id'].apply(str)
    test['test_id'] = test['test_id'].apply(str)

    df_all = pd.concat((train, test))
    df_all['question1'].fillna('', inplace=True)
    df_all['question2'].fillna('', inplace=True)

    print('Creating count vector')
    counts_vectorizer = CountVectorizer(max_features=10000 - 1).fit(
        itertools.chain(df_all['question1'], df_all['question2']))
    other_index = len(counts_vectorizer.vocabulary_)

    X1_train_all = create_padded_seqs(df_all[df_all['id'].notnull()]['question1'])
    y_train_all = df_all[df_all['id'].notnull()]['is_duplicate'].values
    X2_train_all = create_padded_seqs(df_all[df_all['id'].notnull()]['question2'])

    X1_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question1'])
    X2_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question2'])

    X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
        train_test_split(X1_train_all, X2_train_all,
                         y_train_all,
                         stratify=y_train_all,
                         test_size=0.3, random_state=1989)

    feats = []
    model = LSTMModel()
    model.fit(X1_train, X2_train, X1_val, X2_val, y_train, y_val)

    train_features = model.extractFeatures(X1_train_all, X2_train_all)
    test_features = model.extractFeatures(X1_test, X2_test)

    save_data(train_features, y_train_all, args.train_output_file)
    save_data(test_features, None, args.test_output_file)

    features = range(train_features.shape[1])
    feature_map=["lstm_{feature}".format(feature=feature) for feature in features]
