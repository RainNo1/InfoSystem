#-*- coding:utf8 -*-
import math
import os
import pickle
import sys

from base_model.subtopic import Subtopic
from data_prepare.document_preprocessing import doc_preprocessing
from data_prepare.subtopic_preprocessing import  prepare_subtopic,\
    get_subtopic_candidate, get_subtopic_candidate_for_baseline_mining
from util.runname import build_run_name
from util.tf_idf import TF, TF_IDF, IDF
from util.vector_operation import vector_blend


def data_preprocessing(basepath,topicID,sparse_parameters):
    """
    data preprocessing for docs sparse representation.
    ------------
    parameters:
    ------------
    base path : the root path of the data set.
    topicID: the data_preparing id like 0001
    sparse_parameters:
        dictionary_loss:dictionary Optimization Methods("lars","omp","lasso_lars","lasso_cd")
        dictionary_rep: the dictionary original representation(1-tf score addition,2-tf*term_weight )
        dictionary_norm: the dictionary original is normalization or not(Y-yes.N-not) 
        document_ori_rep:the document original representation(D1-tf/idf,D2=tf/idf * bm25,D3-tf/idf * weight of subtopic,D4- bm25 * tf/idf * weight of subtopic)
        document_rep_norm: the document original representation is normalization or not(Y-yes,N-not)
    runname: a file store the middle result for debug.    
    
    ------------
    return:
    ------------
    dictionary_A1: unoptimized dictionary for sparse representation
    documents: a list of docs class
    subtopic_candidates: a list of subtopic class
    word2id_weight: dictionary structure for store the term id and frequency informations
    """


    data_prepare_dict = {}
    runname_list = []
    expansion_methods = sparse_parameters["dictionary_rep"]
    

    for dictionary_norm in sparse_parameters["dictionary_norm"]:
        for dictionary_rep in sparse_parameters["dictionary_rep"]:
            
            for document_ori_rep in sparse_parameters["document_ori_rep"]:
                for document_rep_norm in sparse_parameters["document_rep_norm"]:
                    
                    for mine_method in sparse_parameters["mine_method"]:
                        print topicID,"preparing the data: ",dictionary_rep+dictionary_norm+document_ori_rep+document_rep_norm,mine_method
                        subtopic_candidates,documents,word2id_weight,dictionary_A1,query = construct_data(basepath,topicID,dictionary_norm,dictionary_rep,document_ori_rep,document_rep_norm,mine_method)
                        for dictionary_loss in sparse_parameters["dictionary_loss"]:
                            runname = build_run_name(dictionary_loss,dictionary_norm,dictionary_rep,document_ori_rep,document_rep_norm,mine_method)
                            runname_list.append([runname,dictionary_loss])
                            if not data_prepare_dict.has_key(runname[0:2]+runname[4:]):
                                data_prepare_dict[runname[0:2]+runname[4:]] = [subtopic_candidates,documents,word2id_weight,dictionary_A1,query]
    return runname_list,data_prepare_dict
    
def construct_data(basepath,topicID,dictionary_norm,dictionary_rep,document_ori_rep,document_rep_norm,mine_method = "1"):
    
    dictionary_A1 = []
    sid_dict = {}               # the word of each subtopic for build original dictionary A.
    subtopic_candidates = []    # a list of subtopic class for dictionary learning.
    documents = []              # a list of docs class for docs sparse representation.
    subtopic_word_dict = {}     # a dict store the subtopic ids of a word
    term_weight = {}            # a dict store the term weight.
    query = []                  # a vector for subtopic probability to original query
    pickle_path = basepath+"pickle_data/"+topicID+"-"+mine_method+"/"
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    pickle_file = pickle_path+dictionary_rep+dictionary_norm+document_ori_rep+document_rep_norm
    if os.path.exists(pickle_file) and os.path.isfile(pickle_file):
        infile = open(pickle_file,"rb")
        subtopic_candidates = pickle.load(infile)
        documents  = pickle.load(infile)
        word2id_weight = pickle.load(infile)
        dictionary_A1 = pickle.load(infile)
        query = pickle.load(infile)
        infile.close()
    else:
#         middle_result_path = basepath+'middle/'+topicID+"/"
#         if not os.path.exists(middle_result_path):
#             os.mkdir(middle_result_path)
#         middle_run_result = open(middle_result_path+dictionary_rep+dictionary_norm+document_ori_rep+document_rep_norm+".out","w")
        subtopic_detail = prepare_subtopic(basepath,topicID,[dictionary_rep],mine_method)
        subtopic_dict,word2id_weight,query_dict,subtopic_lenght_ave,topic_words,ori_subtopics ,subtopic_number = subtopic_detail[dictionary_rep]
        docs,document_term_frequency,average_document_len,topic_word_num = doc_preprocessing(basepath,topicID,word2id_weight,topic_words)
        
        # get subtopic id set of a word
        sids = subtopic_dict.keys()
        sids.sort()
        for sid in sids:
            for id ,label,words,frequency,urls in subtopic_dict[sid]:
                for word in words:
                    if subtopic_word_dict.has_key(word) :#and subtopic_word_dict[word].count(sid)==0:
                        subtopic_word_dict[word].append(label)
                    else:
                        subtopic_word_dict[word] = [label]
                        
        # prepare data for construct the original dictionary A1 
        query = [0.0 for i in range(len(sids))]
        for index,sid in enumerate(sids):
            query[index] = query_dict[sid]
            vec = [0.0 for i in range(len(word2id_weight))]
            for id ,label,words,frequency,urls in subtopic_dict[sid]:
                vec_tf_idf = [0.0 for i in range(len(word2id_weight))]
                score_sum = 0.0
                for word in words:
                    tf = words.count(word)*frequency
                    doc_len = len(words)
                    term_fre = None
                    key_fre = 1
                    num_doc = subtopic_number
                    doc_fre = 0
                    for temp, count in word2id_weight[word][2]:
                        doc_fre += count
                    score_tf_idf= 0.0    
                    if dictionary_rep =="S1":
                        score_tf_idf = TF(tf, doc_len, doc_fre, term_fre, key_fre, num_doc, subtopic_lenght_ave)
                    elif dictionary_rep == "S2":  # consider the weight of the word in this subtopic set
                        score_tf_idf = TF(tf, doc_len, doc_fre, term_fre, key_fre, num_doc, subtopic_lenght_ave)*subtopic_word_dict[word].count(label)/(len(subtopic_word_dict[word])+1)
                    else:
                        score_tf_idf = TF(tf, doc_len, doc_fre, term_fre, key_fre, num_doc, subtopic_lenght_ave)
                    score_sum += score_tf_idf
                    word_index = word2id_weight[word][0]
                    vec_tf_idf[word_index] = score_tf_idf
                if dictionary_norm =="Y" and score_sum != 0:
                    vec_tf_idf = [vec_tf_idf[i]/score_sum for i in range(len(word2id_weight))]
                vec = vector_blend(vec,vec_tf_idf)
                true_sprase_vec=[0.0 for i in range(len(sids))]
                true_sprase_vec[index] =1.0#query_dict[sid]
                subtopic = Subtopic()
                subtopic.set_query_frequency(frequency)
                subtopic.set_query_label(label)
                subtopic.set_query_urls(urls)
                subtopic.set_query_vec_tfidf(vec_tf_idf)
                subtopic.set_query_words(words)
                subtopic.set_query_id(id)
                subtopic.set_true_sprase_vec(true_sprase_vec)
                subtopic_candidates.append(subtopic)
                if sid_dict.has_key(index):
                    sid_dict[index].append(subtopic)
                else:
                    sid_dict[index] = []
                    sid_dict[index].append(subtopic)
            vec_sum = sum(vec)        
            if vec_sum != 0 and dictionary_norm =="Y":
                vec = [vec[i]/vec_sum for i in range(len(vec))]
            dictionary_A1.append(vec)
       
        # term weight computing
        for term in document_term_frequency.keys():
            if not word2id_weight.has_key(term):
                continue
            document_term = document_term_frequency[term][0]
            document_fre= len(document_term_frequency[term][1])
            topic_word_count = topic_word_num
           
            if not term_weight.has_key(term):
                topic_num = len(set(subtopic_word_dict[term]))
                topic_num_all = len(sid_dict)
                subtopic_num_term = 0.0
                for temp, count in word2id_weight[word][2]:
                    subtopic_num_term += count
                if document_ori_rep=="D1":
                    term_weight[term] = 1
                elif document_ori_rep=="D2": # the number of subtopic contain this term/the number of subtopic set
                    term_weight[term] = math.log(topic_num_all/(topic_num+1)+1,10)
                elif document_ori_rep=="D3": # the idf score of the term in subtopic collection
                    term_weight[term] = IDF(subtopic_num_term,subtopic_number)
                elif document_ori_rep=="D4":
                    term_weight[term] = math.log(topic_word_count/(len(topic_words)*(document_term+1 )+1),10)
                elif document_ori_rep=="D5":
                    term_weight[term] = 1/(math.log(topic_word_count/(document_fre+1 )+1,2))
        
    
        # preparing docs for sparse representation    
        for doc in docs:
            id = doc.get_id()
            terms = doc.get_term_vec()
    #         true_rank = doc.get_true_rank()
            related_sid = doc.get_related_sid()
            tf_idf_vec = [0.0 for i in range(len(word2id_weight))]
            tf_idf_score_sum = 0.0
#             print >>middle_run_result,id
#             print >>middle_run_result,doc.get_doc_str()
#             print >>middle_run_result,doc.get_related_sid()
            for term in terms:
                tf = terms.count(term)
                doc_len = len(terms)
                doc_fre = 1
                term_fre = None
                num_doc = len(docs)
                key_fre = word2id_weight[term][1]
                score_tf_idf = TF_IDF(tf, doc_len, doc_fre, term_fre, 1, num_doc, average_document_len)*term_weight[term]
                term_index = word2id_weight[term][0]
                tf_idf_vec[term_index] = score_tf_idf
                tf_idf_score_sum += score_tf_idf
            vec = tf_idf_vec
            if document_rep_norm =="Y" and tf_idf_score_sum!=0:
                vec = [tf_idf_vec[i]/tf_idf_score_sum for i in range(len(tf_idf_vec))]
    
            for term in word2id_weight.keys():
                index =  word2id_weight[term][0]
#                 if vec[index]!= 0:
#                     print >>middle_run_result,term,vec[index] ,subtopic_word_dict[term]
            doc.set_tfidf_vec(vec)
            documents.append(doc)
            
        outfile = open(pickle_file,"wb")
        pickle.dump(subtopic_candidates,outfile) 
        pickle.dump(documents,outfile)
        pickle.dump(word2id_weight,outfile)
        pickle.dump(dictionary_A1,outfile)
        pickle.dump(query,outfile)
        outfile.close()
    return subtopic_candidates,documents,word2id_weight,dictionary_A1,query


def data_preprocessing_xquad (basepath,topicID,method):
    """
     data preprocessing for xquad framework.
     ------------
     parameters:
     ------------
        base path : the path of workspace.
        topicID : the original query id.
     ------------
     return:
     ------------
     documents: a list of preparing docs class.
     subtopics: a dictionary of subtopic words.
    """
    subtopicID_word = {}        # a dict of word for the subtopic set
    if method =="mine":
        # mined subtopic collection
        subtopic_list,urlset,word2id_weight,subtopic_length_average,subtopic_term_number_all,topic_words,query_probability_dict,subtopic_number = get_subtopic_candidate_for_baseline_mining(basepath,topicID)

    else:
        # standard subtopic collection
        subtopic_list,urlset,word2id_weight,subtopic_length_average,subtopic_term_number_all,topic_words,query_probability_dict,subtopic_number = get_subtopic_candidate(basepath,topicID)
    docs,document_term_frequency,average_document_len,topic_word_num = doc_preprocessing(basepath,topicID,word2id_weight,topic_words)

    for id,label,words,frequency,urls in subtopic_list:
        for word in words:
            if subtopicID_word.has_key(label):
                subtopicID_word[label].append(word)
            else:
                subtopicID_word[label] = [word]
    return docs,subtopicID_word,query_probability_dict,document_term_frequency,average_document_len,word2id_weight,topic_words
           
def data_preprocessing_mmr (basepath,topicID):
    """
     data preprocessing for mmr framework.
     ------------
     parameters:
     ------------
        base path : the path of workspace.
        topicID : the original query id.
     ------------
     return:
     ------------
     documents: a list of preparing docs class.
     subtopics: a dictionary of subtopic words.
    """
    subtopicID_word = {}        # a dict of word for the subtopic set
    subtopic_list,urlset,word2id_weight,subtopic_length_average,subtopic_term_number_all,topic_words,query_probability_dict,subtopic_number=get_subtopic_candidate(basepath,topicID)
    docs,document_term_frequency,average_document_len,topic_word_num = doc_preprocessing(basepath,topicID,word2id_weight,topic_words)

    for id,label,words,frequency,urls in subtopic_list:
        for word in words:
            if subtopicID_word.has_key(label):
                subtopicID_word[label].append(word)
            else:
                subtopicID_word[label] = [word]
    return docs,subtopicID_word,topic_words,document_term_frequency,average_document_len,word2id_weight
    
if __name__=="__main__":
    print "start..."
    basepath = "D:/diversification/ntcir09/"
    topicID = "0001"
    data_preprocessing_xquad(basepath, topicID)
