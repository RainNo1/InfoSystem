#-*- coding:utf8 -*-
import math


def BM25(tf,doc_length, document_frequency_term,tf_collection, tf_query,ave_document_len,document_num) :
    """
     parameters
     ------------
     tf: the term frequency in the document
     doc_length: the document's length
     document_frequency_term: the document frequency of the term
     tf_collection: the term frequency in the collection
     tf_query: the term frequency in the query
     ave_document_len: the average of document length
     ------------
     return
     ------------
     the score returned by this function .
    """
    # constant parameter 
    k_1 = 1.2
    b = 0.75
    k_3 = 8.0
    #computing the BM25 score
    K = k_1 * ((1 - b) + b * doc_length / ave_document_len) + tf;
   
    score = (tf * (k_3 + 1) * tf_query / ((k_3 + tf_query) * K)) * math.log((document_num - document_frequency_term + 0.5) / (document_frequency_term + 0.5),10)
    print math.log((document_num - document_frequency_term + 0.5) / (document_frequency_term + 0.5),10)
    print (document_num - document_frequency_term + 0.5) / (document_frequency_term + 0.5)
    return score
if __name__=="__main__":
    print "start..."
    print BM25(12 ,164 ,177 ,1539 ,1, 527.015075377 ,199*3)