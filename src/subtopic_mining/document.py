# -*- coding: utf-8 -*-

import copy

from songlib.corpus.countDict import CountDict


class Document:
    def __init__(self):
        self._docStr = ""          # text string
        self._termVec = []         # word segmentation result
        self._posVec = []          # corresponding part of speech
        self._termFrequency = {}   # a dict to record term frequency within the document
        self._featureWeights = {}  # a dict to record feature(term) weightings
        self._label = None         # optional tag for tasks like text classification or sentiment analysis 
        self._id = 0              # 
        
    def __TermCount(self):
        """ A private method to count the term frequencies
        """
        termDict = CountDict()
        for term in self._termVec:
            termDict.AddOne(term)
        return termDict
    
    
    
    def SetId(self, id):
        self._id = id

    def GetId(self):
        return self._id
    
    def SetDocStr(self, docStr):
        self._docStr = docStr

    def GetDocStr(self):
        return self._docStr
    
    def SetTermVec(self, termVec):
        self._termVec = termVec
        self._termFrequency = self.__TermCount()
    
    def GetTermVec(self):
        return self._termVec        
    
    def SetPOSVec(self, posVec):
            self._posVec = posVec    
            
    def GetPOSVec(self):
        return self._posVec
    
    def SetFeatureWeight(self, feature, weight):
        self._featureWeights[feature] = weight
    
    def GetFeatureWeight(self, feature):
        if self._featureWeights.has_key(feature):
            return self._featureWeights[feature]
        return None
    
    def SetFeatureWeights(self, weights):
        self._featureWeights = weights
        
    def GetFeatureWeights(self):
        return self._featureWeights
       
    def GetTermFrequency(self, term):
        if self._termFrequency.has_key(term):
            return self._termFrequency[term]
        return 0

    def SetLabel(self, label):
        self._label = label
        
    def GetLabel(self):
        return self._label

    def __str__(self):
        return self._docStr
    
    def StrGBK(self):
        return self._docStr.decode("utf8", "ignore").encode("gb2312", "ignore")
    
    def DisPlay(self):
        print self._docStr
        print len(self._termVec)
        print len(self._posVec)
        print len(self._termWeightings)   
        
if __name__ == '__main__':
    string = "we love cnu"
    print string.split()
    
    print string + "!!!"
    