# -*- coding:utf-8 -*-
import os
import sys
sys.path.append("..")
from data_prepare.data_preparing import data_preprocessing,\
    data_preprocessing_xquad
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
      
def similarity_computing(documents,subtopics,ave_document_len,query,document_term_frequency):
    """
      computing the document similarity  with the giving topics
      ------------
      parameters
      ------------
      documents: the document collection.
      document: the document which need computing the probability.
      subtopics: the subtopics of the original query.
      ------------
      return
      ------------
    prob_doc_subtopic_vec: Aij represent the similarity of topic i and document j.
    """
   
    prob_doc_subtopic=[] # 0:query,1~n:subtopic
    documents_num = len(documents)
    
    #computing the documents with query related probability
    subtopicID = subtopics.keys()
    subtopicID.append(0)
    subtopicID.sort()
    for sid in range(subtopicID[-1]+1):
        words = []
        vec = [0.0 for i in range(documents_num)]
        if sid == 0:
            words.extend(query)
        elif subtopicID.count(sid)==0:
            prob_doc_subtopic.append([vec,0])
            continue
        else:
            words.extend(subtopics[sid])
        for doc in documents:
            doc_words = doc.get_term_vec()
            doc_length = len(doc_words)
            score_sum = 0.0
            dindex = documents.index(doc)
            for word in words:
                tf = doc_words.count(word)
                tf_collection = 0.0
                document_frequency_term=0
                if document_term_frequency.has_key(word):
                    tf_collection = document_term_frequency[word][0]
                    document_frequency_term = len(document_term_frequency[word][1])
                tf_query = 1
                score_temp = BM25(tf,doc_length,document_frequency_term,tf_collection,tf_query,ave_document_len,documents_num)
                score_sum += score_temp
            vec[dindex] = score_sum
        prob_doc_subtopic.append([vec,sum(vec)])

    return prob_doc_subtopic

   
    
    
    
def best_document_select(documents,subtopicID_word,query,document_term_frequency,average_document_len,word2id_weight,document_selected,prob_doc_subtopic):
    """  """
    alpha = 0.5
    max_score = -100
    max_doc_index = 0
    
    sids = subtopicID_word.keys()
    sids.sort()
    for dindex,document in enumerate(documents):
        if document_selected.count(document)>0:
            continue 
        score_relate = prob_doc_subtopic[0][0][dindex]
        score_diversify = 0.0
        for sid in sids:
            s = 1
            for doc_s in document_selected:
                doc_selected_index = documents.index(doc_s)
                s *= (1-probability_computing(sid, doc_selected_index, prob_doc_subtopic))
            s *= query[sid-1]
            s *= probability_computing(sid,dindex,prob_doc_subtopic)
            score_diversify += s
                
        score = alpha * score_relate + (1-alpha)*score_diversify
        if max_score <score:
            max_score = score
            max_doc_index = dindex
    best_document = documents[max_doc_index]
    best_document.set_ranking_score(max_score)
    return best_document



def xQuAD(basepath,topicID,cutoff= 50):
    """
    return a ranking list using the framework of xQuAD.
    """
    document_selected =[]  #docs ranking list
    
    out_file = open(basepath+"rank/"+topicID+"/xquad_result","w")
    runlist_file = open(basepath+"rank/"+topicID+"/runlist","w")
    print >>runlist_file,"xquad_result"
    runlist_file.close()
    documents,subtopicID_word,query,document_term_frequency,average_document_len,word2id_weight = data_preprocessing_xquad(basepath,topicID)

    sid_doc_related_probility = similarity_computing(documents,subtopicID_word,average_document_len,query,document_term_frequency)
   
    #iteratively selected the best document to build ranking list
    for i in range(cutoff):
        print "%s-th best document."%i
        best_document = best_document_select(documents,subtopicID_word,query,document_term_frequency,average_document_len,word2id_weight,document_selected,sid_doc_related_probility)
        document_selected.append(best_document)
        if len(document_selected) == len(documents):
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
    r = call_eval_for_result(basepath,topicID,10)
    
        
if __name__=="__main__":
    print "start..."
#     basepath = "D:/diversification/ntcir09/"
    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
    for topicID in topics:
        print topicID
        xQuAD(basepath,topicID,30)
    print "Done."