import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from gensim.models import word2vec

pathToBinVectors = 'data/GoogleNews-vectors-negative300.bin'

print "Loading the data file... Please wait..."
from gensim.models import KeyedVectors
model1 = KeyedVectors.load_word2vec_format(pathToBinVectors, binary=True)
print "Successfully loaded 3.6 G bin file!"

# This will return NumPy vector of the word "resume"
# model1['resume']

import numpy as np
import math
from scipy.spatial import distance
from random import sample
import sys
from nltk.corpus import stopwords

class PhraseVector:
    def __init__(self, phrase):
        self.vector = self.PhraseToVec(phrase)

    def __init__(self, phraseArray, dummy):
        self.vector = self.WordArrayToVec(phraseArray)

    # <summary> Calculates similarity between two sets of vectors based on the averages of the sets.</summary>
    # <param>name = "vectorSet" description = "An array of arrays that needs to be condensed into a single array (vector). In this class, used to convert word vecs to phrases."</param>
    # <param>name = "ignore" description = "The vectors within the set that need to be ignored. If this is an empty list, nothing is ignored. In this class, this would be stop words."</param>
    # <returns> The condensed single vector that has the same dimensionality as the other vectors within the vecotSet.</returns>
    def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore=[]):
        if len(ignore) == 0:
            return np.mean(vectorSet, axis=0)
        else:
            return np.dot(np.transpose(vectorSet), ignore) / sum(ignore)

    def PhraseToVec(self, phrase):
        cachedStopWords = stopwords.words("english")
        phrase = phrase.lower()
        wordsInPhrase = [word for word in phrase.split() if word not in cachedStopWords]
        vectorSet = []
        for aWord in wordsInPhrase:
            try:
                wordVector = model1[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    def WordArrayToVec(self, wordArray):
        vectorSet = []
        for aWord in wordArray:
            try:
                wordVector = model1[aWord]
                vectorSet.append(wordVector)
            except:
                pass
        return self.ConvertVectorSetToVecAverageBased(vectorSet)

    # <summary> Calculates Cosine similarity between two phrase vectors.</summary>
    # <param> name = "otherPhraseVec" description = "The other vector relative to which similarity is to be calculated."</param>
    def CosineSimilarity(self, otherPhraseVec):
        cosine_similarity = np.dot(self.vector, otherPhraseVec) / (
        np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
        try:
            if math.isnan(cosine_similarity):
                cosine_similarity = 0
        except:
            cosine_similarity = 0
        return cosine_similarity