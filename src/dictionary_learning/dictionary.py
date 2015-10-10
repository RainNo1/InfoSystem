#-*- coding:utf8 -*-
from numpy.random.mtrand import np
from sklearn.decomposition.dict_learning import SparseCoder

from dictionary_learning.dictionary_learning_cnu import SparseCoder1,\
    dict_learning2
from dictionary_learning.dictionary_learning_cnu import dict_learning1


# from sklearn.decomposition.dict_learning import SparseCoder
# from sklearn.decomposition.dict_learning import dict_learning
# 
class Dictionary():
    # subtopic original representation as TF/IDF vectors
    # dictionary size equal to the length of the subtopic clusters number of the original query 
 
    
    def __init__(self,num_subtopic,train_subtopics,train_subtopics_code,original_A1,dictionary_loss,para_1):
        self.dictionary_size = num_subtopic
        self.train_subtopics = train_subtopics
        self.dictionary_A = original_A1
        self.dictionary_loss = dictionary_loss
        self.coder_n1 = None
        self.coder_n2 = None
        self.coder_l1 = None
        self.coder_l2 = None
        self.coder_l6 = None
        self.coder_l5 = None
        self.para = para_1
        self.train_subtopics_code = train_subtopics_code

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
            self.construct_coder(self.para,4,self.dictionary_loss)
        return self.coder
    def construct_coder(self,method):
#         
        if method == "n1":
            D = self.dictionary_builder()
            self.coder_n1 = SparseCoder1(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
        elif method =="n2":
            D = self.dictionary_builder()
            self.coder_n2 = SparseCoder(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
        elif method =="l13":    
            D = self.dictionary_learning_1()
            self.coder_l1 = SparseCoder1(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
        elif method == "l24":
            D = self.dictionary_learning_2()
            self.coder_l2 = SparseCoder1(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
        elif method == "l57":
            D = self.dictionary_learning_1()
            self.coder_l5 = SparseCoder(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
        elif method == "l68":
            D = self.dictionary_learning_2()
            self.coder_l6 = SparseCoder(dictionary=D, 
                            transform_alpha=self.para, transform_n_nonzero_coefs=3,transform_algorithm=self.dictionary_loss)
            
            
        
       
        

        
        
    def dictionary_learning_2(self):
        """
            use the subtopics as training data learning a dictionary.
        """
        self.dictionary_builder()
        V, U, E = dict_learning2(self.train_subtopics, self.dictionary_size, 1,
                                tol=1e-10, max_iter=1000,
                                method='cd',
                                n_jobs=1,
                                code_init=self.train_subtopics_code,
                                dict_init=self.dictionary_A,
                                verbose=False,
                                random_state=None)
        return U
    def dictionary_learning_1(self):
        """
            use the subtopics as training data learning a dictionary.
        """
        self.dictionary_builder()
        V, U, E = dict_learning1(self.train_subtopics, self.dictionary_size, 1,
                                tol=1e-10, max_iter=1000,
                                method='cd',
                                n_jobs=1,
                                code_init=None,
                                dict_init=None,
                                verbose=False,
                                random_state=None)
        return U
        
    def dictionary_builder(self):

        D = np.empty((len(self.dictionary_A), len(self.dictionary_A[0])))
        for i, vec in enumerate(self.dictionary_A):
            D[i] = vec 
#         D /= np.sqrt(np.sum(D ** 2, axis=1))[:, np.newaxis]
        self.dictionary_A =D
        return D
    
    # without dictionary learning and sparse with CNU Sparse Coding
    def document_sparse_representation_n1(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_n1 == None:
            self.construct_coder("n1")
        sparse_coding = self.coder_n1.transform(vec)
        return sparse_coding
    # without dictionary learning and sparse with sklearn Sparse Coding
    def document_sparse_representation_n2(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_n2 == None:
            self.construct_coder("n2")
        sparse_coding = self.coder_n2.transform(vec)
        return sparse_coding
    #dictionary learning with CNU Learning1 and sparse coding with CNU
    def document_sparse_representation_l13(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_l1 == None:
            self.construct_coder("l13")
        sparse_coding = self.coder_l1.transform(vec)
        return sparse_coding
    #dictionary learning with CNU Learning2 and sparse coding with CNU
    def document_sparse_representation_l24(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_l2 == None:
            self.construct_coder("l24")
        sparse_coding = self.coder_l2.transform(vec)
        return sparse_coding
    #dictionary learning with CNU Learning1 and sparse coding with sk learn
    def document_sparse_representation_l57(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_l5 == None:
            self.construct_coder("l57")
        sparse_coding = self.coder_l5.transform(vec)
        return sparse_coding
    #dictionary learning with CNU Learning2 and sparse coding with sk learn
    def document_sparse_representation_l68(self,document):
        vec = document.get_tfidf_vec()
        if self.coder_l6 == None:
            self.construct_coder("l68")
        sparse_coding = self.coder_l6.transform(vec)
        return sparse_coding
    
    def subtopic_sparse_representation(self,subtopic):
        vec = subtopic.get_query_vec_tfidf()
        if self.coder == None:
            self.construct_coder()
            
        sparse_coding = self.coder.transform(vec)
        return sparse_coding
     
    
        
     
     
     
if __name__=="__main__":
    print "start..."
    d = Dictionary()
    d.dictionary_learning()
    