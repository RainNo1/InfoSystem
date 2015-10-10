#-*- coding:utf8 -*-
import math

class Similarity:
    def __init__(self,docs,words_fre,query,dIndex):
        self.dindex = dIndex
        self.query = query
        self.words_fre_dict = words_fre
        self.num_docs = len(docs)
        self.docs = docs
        self.ave_doc_len =sum([len(doc.GetTermVec()) for doc in docs])/len(docs)
    
    def BM25_score_docs(self):
        score_docs = [0.0 for i in range(self.num_docs)]
        for docid,doc in enumerate(self.docs):
            sim = 0.0
            for wid,word in enumerate(self.query):
                docTermVec = doc.GetTermVec()
                tf = docTermVec.count(word)
                doc_len = len(docTermVec)
                tf_query = 1
                tf_docs = len(self.dindex.GetDocListByKey(word))
                
                tf_all = 0
                
                if self.words_fre_dict.has_key(word):
                    tf_all = self.words_fre_dict[word]
                num_docs = self.num_docs
                ave_doc_len = self.ave_doc_len
              
                sim += self.BM25_score(tf, doc_len, tf_docs, tf_all, tf_query, num_docs, ave_doc_len)
            score_docs[docid] = sim/len(self.query)
        return score_docs
        
    def DHP_score(self,tf,doc_len,tf_docs,tf_all,tf_query,num_docs,ave_doc_len):
        """
            tf: the term frequency in this docs
            doc_len: docs's length
            tf_docs: the documents frequency of the term
            tf_all: the term frequency in the all collection
            tf_query: the term frequency in the query
            num_docs: the number of documents 
            ave_doc_len: the average length of the documents
        """
        f = tf/doc_len
        norm =(1-f)*(1-f)/(tf+1)
        sim = tf_query*norm*(tf* math.log((tf*ave_doc_len/doc_len)* num_docs/tf_all),2)+0.5*math.log(2*math.pi*tf*(1-f),2)
        return sim
    def BM25_score(self,tf,doc_len,tf_docs,tf_all,tf_query,num_docs,ave_doc_len,b=0.75):
        """
            tf: the term frequency in this docs
            doc_len: docs's length
            tf_docs: the documents frequency of the term
            tf_all: the term frequency in the all collection
            tf_query: the term frequency in the query
            num_docs: the number of documents 
            ave_doc_len: the average length of the documents
            b: the parameter of this function 
        """
        #k_l is the parameter of the function BM25,here we choose 1.2
        k_1 = 1.2
        k_3 = 8.0
        sim = 0.0
        
        K = k_1 * ((1 - b) + b * doc_len / ave_doc_len) + tf;
        
        sim = math.log((num_docs + 0.5) / (tf_docs+ 0.5))* ((k_1 + 1.0) * tf / (K + tf)) *((k_3+1)*tf_query/(k_3+tf_query));
 
        return sim
    
    def Hiemstra_LM_score(self,tf,doc_len,tf_docs,tf_all,tf_query,num_tokens):
        """
            tf: the term frequency in this docs
            doc_len: docs's length
            tf_docs: the documents frequency of the term
            tf_all: the term frequency in the all collection
            tf_query: the term frequency in the query
            num_docs: the number of documents 
            ave_doc_len: the average length of the documents
            b: the parameter of this function 
        """    
        c = 0.15
        sim = math.log(1 + (c * tf * num_tokens)/ ((1-c) * tf_all * doc_len))
        return sim  

if __name__=="__main__":
    print "start..."