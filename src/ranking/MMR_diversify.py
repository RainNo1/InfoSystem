#-*- coding:utf8 -*-
import multiprocessing
import os
import sys
sys.path.append("..")
from sklearn.metrics.pairwise import cosine_similarity

from data_prepare.data_preparing import data_preprocessing_mmr
from result.result_analysis import DR_result_analysis
from util.bm25 import BM25
from util.evaluation import call_eval_for_result



def similarity(query,doc,document_term_frequency,average_document_len,document_num):
    """
    computing the similar between query and giving document.
    which was based on the BM25 similar function.
    -----------
    return similar value.
    """
    sim = 0.0
    num = 0
    query_vec = []
    query_vec.extend(query)
    for word in query_vec:
        tf = doc.count(word)
        word = word.split("/")[0]
#         print word
        
        doc_length = len(doc)
        tf_collection = 0.0
        document_frequency_term=0
        if document_term_frequency.has_key(word):
            tf_collection = document_term_frequency[word][0]
            document_frequency_term = len(document_term_frequency[word][1])
        tf_query = query_vec.count(word)
        ave_document_len = average_document_len
#         print tf, doc_length, document_frequency_term, tf_collection, tf_query, ave_document_len, document_num
        sim_word = BM25(tf, doc_length, document_frequency_term, tf_collection, tf_query, ave_document_len, document_num*5)
#         print sim_word
        sim += sim_word
        num+=1
  
    return sim#/num



def similar_documents(documents):
    """
    similar of documents in the collection;
    """
    sim_dict = {}
    for document in documents:
        document_id =document.get_id()
        for doc in documents:
            doc_id = doc.get_id()
            if sim_dict.has_key((document_id,doc_id)) or sim_dict.has_key((doc_id,document_id)):
                continue
            else:
                sim = cosine_similarity(document.get_tfidf_vec(), doc.get_tfidf_vec())
               
                sim_dict[(document_id,doc_id)] = sim[0][0]
    return sim_dict 

def select_best_documnt(docs,doc_selected,sim_dict):
    ""
    best_document_index = -1
    best_document_score = -10000000
    for dindex, document in enumerate(docs):
        document_id = document.get_id()
        if doc_selected.count(document) >0:
            continue
        max_sim = 0.0
        for doc in doc_selected:
            doc_id = doc.get_id()
            if sim_dict.has_key((document_id,doc_id)):
                sim = sim_dict[(document_id,doc_id)]
            elif sim_dict.has_key((doc_id,document_id)):
                sim = sim_dict[(doc_id,document_id)]
            else:
                sim = 0.0
            if max_sim<sim :
                max_sim = sim
        max_sim = -1*max_sim
#         print best_document_score,max_sim
        if best_document_score<max_sim:
            best_document_score = max_sim
            best_document_index = dindex
    if best_document_index == -1:
        for dindex, document in enumerate(docs):
            document_id = document.get_id()
            if doc_selected.count(document) == 0 :
                best_document_index = dindex
                break    
    best_document = docs[best_document_index]
    best_document.set_ranking_score(best_document_score*(-1))
    return best_document  
    

def MMR_ranking(basepath,topicID,cutoff):
    ""
    docs,subtopicID_word,topic_words,document_term_frequency,average_document_len,word2id_weight= data_preprocessing_mmr(basepath,topicID)
    query = topic_words
    first_score = -1000.0
    first_doc_id = -1
    document_collection = []
    doc_selected = []
    #prepare the out files for evaluation
    out_file = open(basepath+"rank/"+topicID+"/mmr_result","w")
    runlist_file = open(basepath+"rank/"+topicID+"/runlist","w")
    print >>runlist_file,"mmr_result"
    runlist_file.close()
    
    
    #select the first document by relevance
    for dindex,doc in enumerate(docs):
        sim = similarity(query,doc.get_term_vec(),document_term_frequency,average_document_len,len(docs))
        sum_weight = 0.0
        vec = [0.0 for i in range(len(word2id_weight))]
        for word in doc.get_term_vec():
            wid = word2id_weight[word][0]
            vec[wid] = word2id_weight[word][1]*1.0
            sum_weight += word2id_weight[word][1]*1.0
        if sum_weight>0:
            doc.set_tfidf_vec([vec[i]/sum_weight for i in range(len(vec))])
        else:
            doc.set_tfidf_vec(vec)
        #doc.set_ranking_score(0.0)
        if first_score< sim:
            first_doc_id = dindex
            first_score = sim
    docs[first_doc_id].set_ranking_score(sim)
    doc_selected.append(docs[first_doc_id])
    sim_dict = similar_documents(docs)
    
    #iteratively selected the best document to build ranking list
    for i in range(cutoff):
        print "%2d-th document selecting for ranking"%(i+1)
        doc_selected.append(select_best_documnt(docs,doc_selected,sim_dict))
        if len(doc_selected) == len(docs):
            # all document in ranking list
            break
        
    # print the ranking result to the evaluation files
    for index,document in enumerate(doc_selected):
       
        documentID = document.get_id()
        document_rank = index+1
        document_score = document.get_ranking_score()
        print >>out_file,topicID +" Q0 "+documentID+" "+str(document_rank) +" "+str(document_score)+ " TESTRUN"
    out_file.close()
    
    # call the evaluation function to evaluate the run result
    r = call_eval_for_result(basepath,topicID,10)    
def run_ranking(basepath,topics):          
    for topicID in topics:
        print topicID
        MMR_ranking(basepath,topicID,30)
if __name__=="__main__":
    print "start..."

    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()

    topics1 = topics
    threads=[]
    for i in range(10):
        ts1 = topics1[i::10]
        t1=  multiprocessing.Process(target=run_ranking,args=([basepath,ts1]))
        threads.append(t1)
                
    for i in range(10):
        threads[i].start()
    for i in range(10):
        threads[i].join()    
 
    DR_result_analysis(basepath,"","mmr_result") 
    print "Done."