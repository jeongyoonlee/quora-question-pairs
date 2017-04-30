import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import distance

def str_jaccard(str1, str2):


    str1_list = str1.split(" ")
    str2_list = str2.split(" ")
    res = distance.jaccard(str1_list, str2_list)
    return res

# shortest alignment
def str_levenshtein_1(str1, str2):


    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=1)
    return res

# longest alignment
def str_levenshtein_2(str1, str2):

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.nlevenshtein(str1, str2,method=2)
    return res

def str_sorensen(str1, str2):

    str1_list = str1.split(' ')
    str2_list = str2.split(' ')
    res = distance.sorensen(str1_list, str2_list)
    return res

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

    print('Generate jaccard')
    train['jaccard'] = train.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
    test['jaccard'] = test.astype(str).apply(lambda x:str_jaccard(x['question1'],x['question2']),axis=1)
    feats.append('jaccard')

    print('Generate porter jaccard')
    train['porter_jaccard'] = train.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)
    test['porter_jaccard'] = test.astype(str).apply(lambda x:str_jaccard(x['question1_porter'],x['question2_porter']),axis=1)

    pd.to_pickle(train[feats].values,args.train_output_file)
    pd.to_pickle(test[feats].values,args.test_output_file)


# print('Generate levenshtein_1')
# train_levenshtein_1 = train.astype(str).apply(lambda x:str_levenshtein_1(x['question1'],x['question2']),axis=1)
# test_levenshtein_1 = test.astype(str).apply(lambda x:str_levenshtein_1(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_levenshtein_1,path+"train_levenshtein_1.pkl")
# pd.to_pickle(test_levenshtein_1,path+"test_levenshtein_1.pkl")

# print('Generate porter levenshtein_1')
# train_porter_levenshtein_1 = train.astype(str).apply(lambda x:str_levenshtein_1(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_levenshtein_1 = test.astype(str).apply(lambda x:str_levenshtein_1(x['question1_porter'],x['question2_porter']),axis=1)

# pd.to_pickle(train_porter_levenshtein_1,path+"train_porter_levenshtein_1.pkl")
# pd.to_pickle(test_porter_levenshtein_1,path+"test_porter_levenshtein_1.pkl")


# print('Generate levenshtein_2')
# train_levenshtein_2 = train.astype(str).apply(lambda x:str_levenshtein_2(x['question1'],x['question2']),axis=1)
# test_levenshtein_2 = test.astype(str).apply(lambda x:str_levenshtein_2(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_levenshtein_2,path+"train_levenshtein_2.pkl")
# pd.to_pickle(test_levenshtein_2,path+"test_levenshtein_2.pkl")

# print('Generate porter levenshtein_2')
# train_porter_levenshtein_2 = train.astype(str).apply(lambda x:str_levenshtein_2(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_levenshtein_2 = test.astype(str).apply(lambda x:str_levenshtein_2(x['question1_porter'],x['question2_porter']),axis=1)

# pd.to_pickle(train_porter_levenshtein_2,path+"train_porter_levenshtein_2.pkl")
# pd.to_pickle(test_porter_levenshtein_2,path+"test_porter_levenshtein_2.pkl")


# print('Generate sorensen')
# train_sorensen = train.astype(str).apply(lambda x:str_sorensen(x['question1'],x['question2']),axis=1)
# test_sorensen = test.astype(str).apply(lambda x:str_sorensen(x['question1'],x['question2']),axis=1)
# pd.to_pickle(train_sorensen,path+"train_sorensen.pkl")
# pd.to_pickle(test_sorensen,path+"test_sorensen.pkl")

# print('Generate porter sorensen')
# train_porter_sorensen = train.astype(str).apply(lambda x:str_sorensen(x['question1_porter'],x['question2_porter']),axis=1)
# test_porter_sorensen = test.astype(str).apply(lambda x:str_sorensen(x['question1_porter'],x['question2_porter']),axis=1)

# pd.to_pickle(train_porter_sorensen,path+"train_porter_sorensen.pkl")
# pd.to_pickle(test_porter_sorensen,path+"test_porter_sorensen.pkl")


