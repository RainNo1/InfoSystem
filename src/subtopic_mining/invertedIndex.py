# -*- coding: utf-8 -*-
from document import Document

class Index(dict): # Index is a dict
    def __init__(self):
        dict.__init__(self)
        self._keyIds = {}
        self._currentId = 1
        self._numOfDocs = 0
        self._docSet = set()
        
    @staticmethod
    def BuildIndex(doclist):
        index = Index()
            
        for docId, doc in enumerate(doclist):
            doc1 = Document()
            doc1 = doc
            # print doc.GetTermVec()
            for word in doc1.GetTermVec():
                index.AddOne(word, docId) #doc1.GetId())
            
        return index       
        
    def AddOne(self, key, docId):
        if docId not in self._docSet:
            self._docSet.add(docId)
            self._numOfDocs += 1
            
        if not self._keyIds.has_key(key):
            self._keyIds[key] = self._currentId
            self._currentId += 1
            
        if self.has_key(key):
            if self[key].has_key(docId):
                self[key][docId] += 1
            else:
                self[key][docId] = 1
        else:
            self[key] = {}
            self[key][docId] = 1
    
    def GetDocListByKey(self, key):
        if self.has_key(key):
            return self[key].keys()
        return [] 
    
    def GetDocHitsListByKey(self, key): # return the list of (docId, freq in document docId)
        if self.has_key(key):
            return self[key].items()
        return []
    
    def NumOfKeys(self):
        return len(self.keys())
    
    def GetKeyIds(self):
        return self._keyIds
    
    def GetNumberOfDocs(self):
        return self._numOfDocs

if __name__ == '__main__':
    print ""