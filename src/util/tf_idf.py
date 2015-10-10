#-*- coding:utf8 -*-
import math

def TF_IDF(tf, docLength, documentFrequency, termFrequency, keyFrequency,numberOfDocuments,averageDocumentLength):
    """
    Uses TF_IDF to compute a weight for a term in a docs.
    
    parameters:
    ------------
     tf:     The term frequency in the docs.
     docLength:     The docs's length.
     termFrequency:     the term frequency in the collection.
     keyFrequency:     the term frequency in the query.
     documentFrequency:     The docs frequency of the term.
     numberOfDocuments:     The number of documents in the collection.
     averageDocumentLength:    The average length of documents in the collection.
     
    return
    ------------
     the score returned by the implemented weighting model.     
    """
    k_1 = 1.2
    b = 0.75
    Robertson_tf = k_1*tf/(tf+k_1*(1-b+b*docLength/averageDocumentLength));
    idf = math.log(numberOfDocuments/documentFrequency+1,10)
    return keyFrequency*Robertson_tf * idf



def TF(tf, docLength, documentFrequency, termFrequency, keyFrequency,numberOfDocuments,averageDocumentLength):
    """
    Uses TF_IDF to compute a weight for a term in a docs.
    
    parameters:
    ------------
     tf:     The term frequency in the docs.
     docLength:     The docs's length.
     termFrequency:     the term frequency in the collection.
     keyFrequency:     the term frequency in the query.
     documentFrequency:     The docs frequency of the term.
     numberOfDocuments:     The number of documents in the collection.
     averageDocumentLength:    The average length of documents in the collection.
     
    return
    ------------
     the score returned by the implemented weighting model.     
    """
    k_1 = 1.2
    b = 0.75
    Robertson_tf = k_1*tf/(tf+k_1*(1-b+b*docLength/averageDocumentLength));
    idf = math.log(numberOfDocuments/documentFrequency+1,10)
    return keyFrequency*Robertson_tf 

def IDF(number_documents,document_frequency):
    """
    """
    idf = math.log(number_documents/(document_frequency+1)+1,10)
    return idf
if __name__=="__main__":
    print "start..."