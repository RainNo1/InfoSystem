#-*- coding:utf8 -*-

class Subtopic():
    
    
    def init(self):
        self.query_id = None
        self.query_words=None
        self.query_vec_onehot = None
        self.query_vec_tfidf = None
        self.query_frequency = None
        self.query_urls = None
        self.query_label = None
        self.true_sprase_vec = None
        self.run_sprase_vec = None

    def get_query_words(self):
        return self.query_words

    def get_query_id(self):
        return self.query_id

    def get_query_vec_onehot(self):
        return self.query_vec_onehot

    def get_query_vec_tfidf(self):
        return self.query_vec_tfidf

    def get_query_frequency(self):
        return self.query_frequency

    def get_query_urls(self):
        return self.query_urls

    def get_query_label(self):
        return self.query_label

    def get_true_sprase_vec(self):
        return self.true_sprase_vec

    def get_run_sprase_vec(self):
        return self.run_sprase_vec
    
    def set_query_id(self,id):
        self.query_id = id
        
    def set_query_words(self, words):
        self.query_words = words

    def set_query_vec_onehot(self, onehot_vector):
        self.query_vec_onehot = onehot_vector

    def set_query_vec_tfidf(self, tfidf_vector):
        self.query_vec_tfidf = tfidf_vector

    def set_query_frequency(self, frequency):
        self.query_frequency = frequency

    def set_query_urls(self, urls):
        self.query_urls = urls

    def set_query_label(self, label):
        self.query_label = label

    def set_true_sprase_vec(self, true_sprase_vec):
        self.true_sprase_vec = true_sprase_vec

    def set_run_sprase_vec(self, run_sprase_vec):
        self.run_sprase_vec = run_sprase_vec
    def display (self):
        s =sum(self.get_run_sprase_vec())
        
        print "subtopic:%s"% self.get_query_id()
        print "true label:%s" % self.get_query_label()
        print "sparse representation:%s"%self.get_run_sprase_vec()#str([self.get_run_sprase_vec()[i]/s for i in range(len(self.get_run_sprase_vec()))])
    
if __name__=="main":
    print "start..."