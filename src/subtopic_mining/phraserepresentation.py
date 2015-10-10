from wordrepresentation import WordRepresentation

def VectorBlend(vectors, weights):
    #print vectors
    #print weights
    n = len(vectors)
    if n > 0: 
        if n == 1:
            return vectors[0]
        dim = len(vectors[0])
        vec = [0.0 for d in range(dim)]
                
        for i in range(n):
            for d in range(dim):
                vec[d] += vectors[i][d] * weights[i]
        return vec
    return None

class PhraseRepresentation(object):
    def __init__(self, label, string, wordrep_list,wordlist, weights, originalquery,url,emworddict,iIndex,subtopic_frequency):
        self.label = label
        self.iIndex = iIndex
        self.emworddict = emworddict
        self.phrasestr = string
        self.ori_query = originalquery
        self.wordreps = wordrep_list
        self.wordlist = wordlist
        self.wordweights = weights
        self.frequency = subtopic_frequency
        self.url = url
        self.URLVec = None
        self.vsm = None
        self.categoryVec = None
        self.embeddingVec = None
        self.invertedDocVec = None
        self.topicVec = None
        self.embeddingWordVec = None

    def _constructVSM(self):
        onehotvecs = []
        for word in self.wordreps:
            onehotvecs.append(word.GetOneHotVec())
        self.vsm = VectorBlend(onehotvecs, self.wordweights)

    def GetVSM(self):
        if self.vsm == None:
            self._constructVSM()
        return self.vsm

    def _constructCategoryVec(self):
        wordstrs = []
        for word in self.wordreps:
            wordstrs.append(word.str)
        query = self.ori_query + " " + " ".join(wordstrs)
        #print "new query", query
        categorylist = WordRepresentation.queryclassifier.Predict(query)
        self.categoryVec = [item[1] for item in categorylist]
        #print query, categorylist
        #catevecs = []
        #for word in self.wordreps:
        #    catevecs.append(word.GetCategoryVec())
        #self.categoryVec = VectorBlend(catevecs, self.wordweights)

    def GetCategoryVec(self):
        if self.categoryVec == None:
            self._constructCategoryVec()
        return self.categoryVec

    def _constructEmbeddingVec(self):
        embeddingvecs = []
        weights = []
        weight_sum = 0.0
        for i, word in enumerate(self.wordreps):
            wvec = word.GetEmbeddingVec()
            if wvec != None:
                embeddingvecs.append(wvec)#word.GetEmbeddingVec())
                weights.append(self.wordweights[i])
                weight_sum += self.wordweights[i]
        weights = [weight / weight_sum for weight in weights]
        self.embeddingVec = VectorBlend(embeddingvecs, weights)

    def GetEmbeddingVec(self):
        if self.embeddingVec == None:
            self._constructEmbeddingVec()
        return self.embeddingVec

    def _constructInvertedDocVec(self):
        idocvecs = []
        for word in self.wordreps:
            idocvecs.append(word.GetInvertedDocVec())
        self.invertedDocVec = VectorBlend(idocvecs, self.wordweights)
    
    def GetInvertedDocVec(self):
        if self.invertedDocVec == None:
            self._constructInvertedDocVec()
        return self.invertedDocVec
    def _constructURLVec(self):
#         idocvecs = []
#         for word in self.wordreps:
#             idocvecs.append(word.GetInvertedDocVec())
#         self.URLVec = VectorBlend(idocvecs, self.wordweights)
#         self.URLVec.extend(self.url)
        self.URLVec=self.url
    def _constructEmbeddingWordVec(self):
        emwordvecs = []
        weights = []
        wsum = 0.0
 
        for i, word in enumerate(self.wordreps):
            csum = 0.0
            vec = [0.0 for j in range(len(self.emworddict))]
            wordweights =  word.GetEmbeddingwordlist()
            
            if self.emworddict.has_key(word.str):
                
                id = self.emworddict[word.str]
                count = len(self.iIndex.GetDocListByKey(word.str))*1.0
                
                if not wordweights == None:
                    for term in wordweights:
                        c = len(self.iIndex.GetDocListByKey(term))*1.0
                        if self.emworddict.has_key(term):
                            vec[self.emworddict[term]] += c
                            csum += c
                        else:
                            count += c
                        
                vec[id] = count
                csum += count
            else:
                count = len(self.iIndex.GetDocListByKey(word.str))*1.0
                if not wordweights == None:
                    for term in wordweights:
                        c = count + len(self.iIndex.GetDocListByKey(term))*1.0
                        if self.emworddict.has_key(term):
                            vec[self.emworddict[term]] += c
                            csum += c
            if csum != 0.0:  
                vec = [count/csum for count in vec]
            emwordvecs.append(vec)
            weights.append(self.wordweights[i])
            wsum += self.wordweights[i]
          
        weights = [weight/wsum for weight in weights]
        self.embeddingWordVec=VectorBlend(emwordvecs, weights)
    
    def GetURLVec(self):
        if self.URLVec == None:
            self._constructURLVec()
        return self.URLVec
    def GetEmbeddingWordVec(self):
        if self.embeddingWordVec == None:
            self._constructEmbeddingWordVec()
        return self.embeddingWordVec
    
    def GetTopicVec(self):
        pass
    
    def GetWordlist(self):
        return self.wordreps