# -*- coding: utf-8 -*-
import sys
sys.path.append("/users/songwei/query_classification/queryclassification/")
from queryclassification import QueryClassifier, label_category59, label_category10
from gensim.models.word2vec import Word2Vec

class WordRepresentation(object):
#     model = Word2Vec.load_word2vec_format('/users/songwei/softwares/word2vec/vectors300_5.bin', binary=True)
#     queryclassifier = QueryClassifier(modelfile="/users/songwei/query_classification/SVM/new_59cate_bigram.model", dictfile="/users/songwei/query_classification/SVM/new_59cate_bigram.dict", label_category=label_category59)
#     queryclassifier = QueryClassifier(modelfile="/users/songwei/query_classification/SVM/new_10cate_bigram.model", dictfile="/users/songwei/query_classification/SVM/new_10cate_bigram.dict", label_category=label_category10)
    def __init__(self, wordStr=None, w2id=None, docIndex=None):
        self.str = wordStr
        self.word2id = w2id
        self.docIndex = docIndex
        self.simwordlist = None
        self.onehotRep = None
        self.invertedDocRep = None
        self.topicModelRep = None
        self.categoryRep = None
        self.embeddingRep = None
        self.onehot_dim = len(self.word2id.keys()) if self.word2id!=None else 0
        self.embeddingwordlist = None

    def _constructOneHotVec(self):
        vec = [0.0 for i in range(self.onehot_dim)]
        wid = self.word2id[self.str][0]
        vec[wid] = 1.0
        self.onehotRep = vec

    def GetOneHotVec(self):
        if self.onehotRep == None:
            self._constructOneHotVec()
        return self.onehotRep

    def _constructInvertedDocVec(self):
        vector = [0.0 for i in range(self.docIndex.GetNumberOfDocs())]
        doclist = self.docIndex.GetDocListByKey(self.str)
        for id in doclist:
            vector[id] = 1.0
        self.invertedDocRep = vector
        
    def GetInvertedDocVec(self):
        if self.invertedDocRep == None:
            self._constructInvertedDocVec()
        return self.invertedDocRep

    def _constructEmbeddingVec(self):
        embedding = None
        try:
            embedding = WordRepresentation.model[self.str] # numpy.narray type
        except:
            pass
        self.embeddingRep = embedding

    def GetEmbeddingVec(self):
        if self.embeddingRep == None:
            self._constructEmbeddingVec()
        return self.embeddingRep

    def _constructCategoryVec(self):
        categorydist = WordRepresentation.queryclassifier.Predict(self.str)
        self.categoryRep = [item[1] for item in categorydist]

    def GetCategoryVec(self):
        if self.categoryRep == None:
            self._constructCategoryVec()
        return self.categoryRep
    #return: [(word, weight)]
    def GetSimilarWordWeights(self,wordlist = None,top_n= 50):#wordlist=[self.str],
        wordweights = None
        if wordlist == None:
            try:
                wordweights = WordRepresentation.model.most_similar(positive=[self.str], topn=50)
            except:
                pass
        else:
            try:
                wordweights = WordRepresentation.model.most_similar(positive=wordlist, topn=top_n)
            except:
                pass
        return wordweights
    def SetEmbeddingwordlist(self,wordlist):
        self.embeddingwordlist = wordlist
    def GetEmbeddingwordlist(self):
        return self.embeddingwordlist