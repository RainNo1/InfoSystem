#-*- coding:utf8 -*-
# from gensim.models.word2vec import Word2Vec
from subtopic_mining.wordrepresentation import WordRepresentation


class WordEmbeddingFramework():
    """
    a class load the word embedding model,to get the word embedding vector and similar words.
    """
    def __init__(self):
        ""
        self.model  = WordRepresentation()


    # for the giving words, get the Top 30 expansion words for it
    def get_expansion_words(self,words):
        expansion_words = None

        try:
            expansion_words = self.model.model.most_similar(positive=words, topn=15)
        except:
            pass
        
        return expansion_words
    
    # get the embedding vector for giving word
    def get_embedding_vector(self,word):
        embedding_vec = None
        try:
            embedding_vec = self.model.model[word]
        except:
            pass
        
        return embedding_vec
if __name__=="__main__":
    print "start..."