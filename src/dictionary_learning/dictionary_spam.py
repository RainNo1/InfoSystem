#-*- coding:utf8 -*-
from numpy.random.mtrand import np
from sklearn.decomposition.dict_learning import SparseCoder


class Dictionary_spam():
    # subtopic original representation as TF/IDF vectors
    # dictionary size equal to the length of the subtopic clusters number of the original query 
 
    
    def __init__(self,num_subtopic,subtopic_candidates,original_A1):
        self.dictionary_size = num_subtopic
        self.subtopic_candidates = subtopic_candidates
        self.dictionary_A = original_A1
        self.coder = None

    def get_dictionary_size(self):
        return self.dictionary_size


    def get_subtopic_candidates(self):
        return self.subtopic_candidates


    def get_dictionary_a(self):
        return self.dictionary_A


    def set_dictionary_size(self, value):
        self.dictionary_size = value


    def set_subtopic_candidates(self, value):
        self.subtopic_candidates = value


    def set_dictionary_a(self, value):
        self.dictionary_A = value
        
    def get_coder(self):
        if self.coder == None:
            self.construct_coder(1,4,"lars")
        return self.coder
    def construct_coder(self,alpha = 0.5,n_nonzero = 4,algo = "lars"):
        D = self.dictionary_builder()
        self.coder = SparseCoder(dictionary=D, transform_n_nonzero_coefs=3,
                            transform_alpha=alpha, transform_algorithm="threshold")
        

        
        
    def dictionary_learning(self):
        """
            use the subtopics as training data learning a dictionary.
        """
        print" dict "
        
    def dictionary_builder(self):
        D = np.empty((len(self.dictionary_A), len(self.dictionary_A[0])))
        for i, vec in enumerate(self.dictionary_A):
            D[i] = vec 
        D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
       
        return D
    
    
    def document_sparse_representation(self,document):
        vec = document.get_tfidf_vec()
        if self.coder == None:
            self.construct_coder()
        sparse_coding = self.coder.transform(vec)
        return sparse_coding
    
    def subtopic_sparse_representation(self,subtopic):
        vec = subtopic.get_query_vec_tfidf()
        if self.coder == None:
            self.construct_coder()
            
        sparse_coding = self.coder.transform(vec)
        return sparse_coding
     

if __name__=="__main__":
    print "start..."