# -*- coding:utf-8 -*-
import multiprocessing
import os
import sys

sys.path.append("..")
from data_prepare.data_preparing import data_preprocessing,\
    data_preprocessing_xquad
from result.result_analysis import DR_result_analysis_sparse
from util.bm25 import BM25
from util.evaluation import call_eval_for_result







def probability_computing(subtopicID,document_index,prob_doc_subtopic_vec):
    """
      imitate the search engine for compute the P(di|qi)
      ------------
      parameters
      ------------
      documents: the document collection.
      document: the document which need computing the probability.
      subtopics
    """
    dindex = document_index
    score_sum = 0.0
    #score_sum = prob_doc_subtopic_vec[subtopicID][1]
    for i in range(len(prob_doc_subtopic_vec)):
        score_sum += prob_doc_subtopic_vec[i][1]
    if score_sum == 0:
        score_sum = 1.0
    probability = prob_doc_subtopic_vec[subtopicID][0][dindex]/score_sum
    
    return probability
      


def document_subtopic_relevance(subtopics,documents,ave_document_len,query_words,document_term_frequency):
    """
      computing the document relevance  with the giving topics
      ------------
      parameters
      ------------
      documents: the document collection.
      document: the document which need computing the probability.
      subtopics: the subtopics of the original query.
      ------------
      return:
        prob_doc_subtopic: a dictionary construct like document_id: [document,relevance2query,relevance2subtopic_vec,relevance_sum].
    """
    prob_doc_subtopic={}
    documents_num = len(documents)
    subtopicIDs = subtopics.keys()
    subtopicIDs.sort()
#     out = open("test1","w")   
    for document in documents:
        document_id = document.get_id()
        # calculating the relevance between document and original query.
        doc_query_rel = 0.0
        doc_words = document.get_term_vec()
        doc_length = len(doc_words)
        
        for word in query_words:
            word = word.split("/")[0]
            tf = 1
            tf_collection = 0.0
            document_frequency_term=0
            if document_term_frequency.has_key(word):
                tf_collection = document_term_frequency[word][0]
                document_frequency_term = len(document_term_frequency[word][1])     
                 
                doc_query_rel += BM25(tf,doc_length,document_frequency_term,tf_collection,tf,ave_document_len,documents_num*1.5)
        doc_query_rel /= len(query_words) #average score of the BM25 to represent the relevance
        
        # calculating the subtopic coverage
        document_subtopic_coverage = [0.0 for i in range(len(subtopicIDs))]
        score_sum = 0.0
        for sindex,sid in enumerate(subtopicIDs): 
            words = []
            words.extend(subtopics[sid])
            score = 0.0
            word_num =len(words)
            for word in words:
               
                tf = doc_words.count(word)
                tf_collection = 0.0
                document_frequency_term=0
                if document_term_frequency.has_key(word):
                    tf_collection = document_term_frequency[word][0]
                    document_frequency_term = len(document_term_frequency[word][1])
                tf_query = 1
#                 print >>out,tf,doc_length,document_frequency_term,tf_collection,tf_query,ave_document_len,documents_num
                score_temp = BM25(tf,doc_length,document_frequency_term,tf_collection,tf_query,ave_document_len,documents_num)
#                 if score_temp<0:
#                     score_temp *= -1
                score_sum += score_temp
                score += score_temp
                if score_temp==0:
                    word_num -= 1
            if word_num>0:
                score /= word_num
            else:
                score = 0.0
            document_subtopic_coverage[sindex] = score
        if not prob_doc_subtopic.has_key(document_id):
            prob_doc_subtopic[document_id]=[document]
            prob_doc_subtopic[document_id].append(doc_query_rel)
            prob_doc_subtopic[document_id].append(document_subtopic_coverage)
            prob_doc_subtopic[document_id].append(sum(document_subtopic_coverage))
         
    #processing the p(d|qi) for calculating the document coverage of subtopic    
    document_subtopic_vec = [0.0 for i in range(len(subtopicIDs))]
    rel_sum_all = 0.0
    for document_id in prob_doc_subtopic.keys():  
        document,dqrelevance,dsrelevance_vec,rel_sum  = prob_doc_subtopic[document_id]
#         print>>out,dsrelevance_vec,rel_sum,dqrelevance
        for sindex,sid in enumerate(subtopicIDs):
            document_subtopic_vec[sindex] += dsrelevance_vec[sindex]
        rel_sum_all += rel_sum
    vec = [document_subtopic_vec[i]/len(documents) for i in range(len(document_subtopic_vec))]
    
    
    return prob_doc_subtopic,vec            
    
    

def select_best_document(document_subtopic_rel,rel_subtopic_ave,query_prob,subtopicID_word,document_selected,filem):
    alpha = 0.5
    max_score = -100000000
    best_document=None
    best_rel = 0.0
    best_div= 0.0
    sids = subtopicID_word.keys()
    sids.sort()
    

    for document_id in document_subtopic_rel.keys():
        document,dqrelevance,dsrelevance_vec,rel_sum = document_subtopic_rel[document_id]    
        if document_selected.count(document)>0:
            continue
        score_relate = dqrelevance     
        score_diversify = 0.0
#         print>>filem,document_id
        if len(document_selected)>0:
            for sindex,sid in enumerate(sids):
                if rel_subtopic_ave[sindex]==0:
                    continue
                s = 1
                for doc_select in document_selected:
                    doc_select_id = doc_select.get_id()
                    d,dq,dsr,rs  = document_subtopic_rel[doc_select_id]
                    s *= (1-dsr[sindex])#/rel_subtopic_ave[sindex])
                    
#                 print "qeury", query_prob[sid]
#                 print "div",dsrelevance_vec[sindex]
                s *= query_prob[sid]
                s *= dsrelevance_vec[sindex]##/rel_subtopic_ave[sindex]
#                 print>>filem,"ddd",query_prob[sid],s
                score_diversify += s
#             print>>filem,"score_diversify:",score_diversify

        score = (1-alpha) * score_relate + alpha*score_diversify
#         print  max_score ,score
        if max_score <score:
            max_score = score
            document.set_ranking_score(max_score)
            best_div = alpha*score_diversify
            best_rel = (1-alpha) * score_relate
            best_document = document
#     print>>filem,"score_diversify:",best_div,best_rel,max_score,best_document.get_true_rank()
    if best_document == None:
        for document_id in document_subtopic_rel.keys():
            document,dqrelevance,dsrelevance_vec,rel_sum = document_subtopic_rel[document_id]   
            if document_selected.count(document)==0:
                best_document = document
                best_document.set_ranking_score(0.0333)
                break
            
    return best_document



def xQuAD(basepath,topicID,filename,cutoff= 50,method="mine"):
    """
    return a ranking list using the framework of xQuAD.
    """
    document_selected =[]  
    
    out_file = open(basepath+"rank/"+topicID+"/"+filename,"w")
    runlist_file = open(basepath+"rank/"+topicID+"/runlist","w")
    print >>runlist_file,"xquad_mine_less"
    print >>runlist_file,"xquad_mine_all"
    print >>runlist_file,"xquad_standard_less"
    print >>runlist_file,"xquad_standard_all"
    runlist_file.close()

    documents,subtopicID_word,query_prob,document_term_frequency,average_document_len,word2id_weight,topic_words = data_preprocessing_xquad(basepath,topicID,method)
    subtopic_word={}
    keys = subtopicID_word.keys()
    keys.sort()
    for key in keys:
        if filename == "xquad_mine_less" or filename =="xquad_standard_less":
            if key==keys[-1]:continue
            if key==keys[-2]:continue
#         if key == 3:continue
        if not subtopic_word.has_key(key):
            subtopic_word[key] = subtopicID_word[key]
#     print len(subtopicID_word.keys()),len(subtopic_word.keys())
    if filename == "xquad_mine_less" or filename =="xquad_standard_less": 
        prob_sum = 0.0
        for key in query_prob.keys():
            if key == keys[-1]:
                query_prob[key]=0
            elif key == keys[-2] and len(query_prob.keys())>3:
                query_prob[key]=0
            else:
                prob_sum+=query_prob[key]
        for key in query_prob.keys():
#             print key,query_prob[key]
            query_prob[key] /= prob_sum
            
    document_subtopic_rel,rel_subtopic_ave = document_subtopic_relevance(subtopic_word,documents,average_document_len,topic_words,document_term_frequency)
    filem= open("middle/"+topicID+"div","w")
    #iteratively selected the best document to build ranking list
    for i in range(cutoff):
        print topicID,"%2d-th best document."%(i+1)
        best_document =  select_best_document(document_subtopic_rel,rel_subtopic_ave,query_prob,subtopic_word,document_selected,filem)
        document_selected.append(best_document)
        if len(document_selected) == len(documents)-1:
            # all document in ranking list
            break
        
    # print the ranking result to the evaluation files
    for index,document in enumerate(document_selected):
        documentID = document.get_id()
        document_rank = index+1
        document_score = document.get_ranking_score()
        print >>out_file,topicID +" Q0 "+documentID+" "+str(document_rank) +" "+str(document_score)+ " TESTRUN"
    out_file.close()
        
    # call the evaluation function to evaluate the run result
#     r = call_eval_for_result(basepath,topicID,10)
    
    
def run_ranking(basepath,topics,methods):    
    
    for topicID in topics:
        for method in methods:
            if method =="mine":
                filename1= "xquad_mine_less"
                filename2 = "xquad_mine_all"
            else:
                filename1= "xquad_standard_less"
                filename2 = "xquad_standard_all"
            
            xQuAD(basepath,topicID,filename1,30,method)
            xQuAD(basepath,topicID,filename2,30,method)
        r = call_eval_for_result(basepath,topicID,10)     
def sysrun( topics1,basepath):
    threads=[]
    for i in range(10):
        ts1 = topics1[i::10]
        t1=  multiprocessing.Process(target=run_ranking,args=([basepath,ts1,["standard","mine"]]))
        threads.append(t1)    
    for i in range(10):
        threads[i].start()
    for i in range(10):
        threads[i].join()
        
if __name__=="__main__":
    print "start..."
#     basepath = "D:/diversification/ntcir09/"
    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
#     run_ranking(basepath,["0243"])
#     raise
  
    sysrun( topics,basepath)
    
    DR_result_analysis_sparse(basepath,"","xquad_mine_less","ntcir09") 
    DR_result_analysis_sparse(basepath,"","xquad_mine_all","ntcir09") 
    DR_result_analysis_sparse(basepath,"","xquad_standard_less","ntcir09") 
    DR_result_analysis_sparse(basepath,"","xquad_standard_all","ntcir09") 
    
    basepath = "/users/songwei/xuwenbin/diversification/ntcir10/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
    sysrun( topics,basepath)
    
    DR_result_analysis_sparse(basepath,"","xquad_mine_less","ntcir10") 
    DR_result_analysis_sparse(basepath,"","xquad_mine_all","ntcir10") 
    DR_result_analysis_sparse(basepath,"","xquad_standard_less","ntcir10") 
    DR_result_analysis_sparse(basepath,"","xquad_standard_all","ntcir10") 
    print "Done."