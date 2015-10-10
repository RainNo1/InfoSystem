# -*- coding:utf-8 -*-

class Document:

    
    def _init_(self):
        self._docStr = ""          # text string
        self._termVec = []         # word segmentation result
        self._posVec = []          # corresponding part of speech
        self._termFrequency = {}   # a dict to record term frequency within the docs
        self._id = ''              # docs filename
        self._sparse_rep=[]        # a vector representation for docs using sparse coding 
        self._related_sid=""       # a vector contain the subtopic id which docs related
        self._true_rank = ""       # integer indicate the index of this docs in ideal ranking list   
        self._tfidf_vec = []       # the original tf-idf vector
        self._ranking_score = 0.0

    def get_doc_str(self):
        return self._docStr


    def get_term_vec(self):
        return self._termVec


    def get_pos_vec(self):
        return self._posVec


    def get_term_frequency(self):
        return self._termFrequency


    def get_id(self):
        return self._id


    def get_sparse_rep(self):
        return self._sparse_rep


    def get_related_sid(self):
        return self._related_sid


    def get_true_rank(self):
        return self._true_rank


    def get_tfidf_vec(self):
        return self._tfidf_vec


    def set_doc_str(self, value):
        self._docStr = value


    def set_term_vec(self, value):
        self._termVec = value


    def set_pos_vec(self, value):
        self._posVec = value


    def set_term_frequency(self, value):
        self._termFrequency = value


    def set_id(self, value):
        self._id = value


    def set_sparse_rep(self, value):
        self._sparse_rep = value


    def set_related_sid(self, value):
        self._related_sid = value


    def set_true_rank(self, value):
        self._true_rank = value


    def set_tfidf_vec(self, value):
        self._tfidf_vec = value

    def get_ranking_score(self):
        return self._ranking_score


    def set_ranking_score(self, value):
        self._ranking_score = value

    
    def _str_(self):
        return self._docStr
    
    def StrGBK(self):
        return self._docStr.decode("utf8", "ignore").encode("gb2312", "ignore")
    
    def display(self):
#         print self.get_doc_str()
#         print self.get_tfidf_vec()
        #s = sum(self.get_sparse_rep())
        print "docs:            %s"% self.get_id()
        print "related subtopic id: %s"%self.get_related_sid()
        print "sparse coding:       %s"% self.get_sparse_rep()#str([self.get_sparse_rep()[i]/s for i in range(len(self.get_sparse_rep()))])
        
    
   
if __name__=="__main__":
    pass