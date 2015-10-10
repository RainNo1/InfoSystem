#-*- coding:utf8 -*-


import os

import numpy

from base_model.document import Document
from data_prepare.data_preparing import data_preprocessing
from dictionary_learning.dictionary import Dictionary
import numpy as np
from util.vector_operation import vector_multiplier


def sparse_representation(basepath,topicID,sparse_parameters):
    """
    """

    sparse_representation_data = []   
        
    runname_list,data_prepare_dict =data_preprocessing(basepath,topicID,sparse_parameters)
    for subtopic_less in sparse_parameters["subtopic_less"]:
        if subtopic_less == "N":
            for para_1 in sparse_parameters["para_1"]:
                for dict_learning in sparse_parameters["dictionary_learning"]:
                    for runname,dictionary_loss in runname_list:
                        runname = runname+str(int(para_1*10)).zfill(2)+"LN"+dict_learning
                        print topicID,"sparse representation: "+runname
                        subtopic_candidates,documents,word2id_weight,dictionary_A1,query = data_prepare_dict[runname[0:2]+runname[4:10]]
                    
                        train_subtopics = np.empty((len(subtopic_candidates), len(dictionary_A1[0])))
                        train_subtopics_code = np.empty((len(subtopic_candidates), len(dictionary_A1)))
                        subtopic_words ={}
                        for index,subtopic in enumerate(subtopic_candidates):
                            train_subtopics[index]=subtopic.get_query_vec_tfidf()
                            train_subtopics_code[index]=subtopic.get_true_sprase_vec()
                            cid =  subtopic.get_query_label()
                            if subtopic_words.has_key(cid):
                                subtopic_words[cid].extend(subtopic.get_query_words())
                            else:
                                subtopic_words[cid]=[]
                                subtopic_words[cid].extend(subtopic.get_query_words())
                        # build the sparse coding dictionary 
                        num_subtopic = len(dictionary_A1)
                        original_A1 = dictionary_A1
            #             print original_A1
                        dictionary = Dictionary(num_subtopic,train_subtopics,train_subtopics_code,original_A1,dictionary_loss,para_1)
#                         middle_result_path = basepath+'middle/'+topicID+"/"
#                         if not os.path.exists(middle_result_path):
#                             os.mkdir(middle_result_path)
#                          
#                         middle_sparse_result = open(middle_result_path+runname+".out","w")
#                         for key,items in subtopic_words.items():
#                             print >>middle_sparse_result,key
#                             words = []
#                             for word in items:
#                                 if word2id_weight.has_key(word):
#                                     words.append(word+"+"+str(word2id_weight[word][1]))
#                             print >>middle_sparse_result,"\t".join(words)
                        doc_rep = []
                        for document in documents:
                            if dict_learning== "N1"or dict_learning=="N3":
                                sparse_coding = dictionary.document_sparse_representation_n1(document)
                            elif dict_learning == "N2" or dict_learning=="N4":
                                sparse_coding = dictionary.document_sparse_representation_n2(document)
                            elif dict_learning == "L1" or dict_learning == "L3":
                                sparse_coding = dictionary.document_sparse_representation_l13(document)
                            elif dict_learning == "L2" or dict_learning == "L4":
                                sparse_coding = dictionary.document_sparse_representation_l24(document)         
                            elif dict_learning == "L5" or dict_learning == "L7":
                                sparse_coding = dictionary.document_sparse_representation_l57(document)
                            elif dict_learning == "L6" or dict_learning == "L8":
                                sparse_coding = dictionary.document_sparse_representation_l68(document)
                            
                            if len(sparse_coding)==1:
                                document.set_sparse_rep(sparse_coding[0])
                            else:
                                document.set_sparse_rep(sparse_coding)
                            rep = document.get_sparse_rep()
                            doc = Document()
                            doc.set_id(document.get_id())
                            doc.set_doc_str(document.get_doc_str())
                            doc.set_related_sid(document.get_related_sid())
                            if dict_learning == "N1" or dict_learning == "N2" or dict_learning == "L1" or dict_learning == "L2" or dict_learning == "L5" or dict_learning == "L6": 
                                doc.set_sparse_rep([rep[i] if rep[i]>0 else 0 for i in range(len(rep))])
                            else:
                                doc.set_sparse_rep(rep)
                            doc.set_term_vec(document.get_term_vec())
                            doc.set_tfidf_vec(document.get_tfidf_vec())
                            doc.set_true_rank(document.get_true_rank())
                            doc.set_pos_vec(document.get_pos_vec())
                            doc_rep.append(doc)
                        sparse_representation_data.append([runname,doc_rep,query])
        else:
            for para_1 in sparse_parameters["para_1"]:
                for dict_learning in sparse_parameters["dictionary_learning"]:
                    for runname,dictionary_loss in runname_list:
                        runname = runname+str(int(para_1*10)).zfill(2)+"LY"+dict_learning
                        print topicID,"sparse representation: "+runname
                        subtopic_candidates,documents,word2id_weight,dictionary_A1,query = data_prepare_dict[runname[0:2]+runname[4:10]]
                        flag= True
                        # build the sparse coding dictionary 
                        if len(dictionary_A1)>3:
                            flag =False
                            original_A1 = dictionary_A1[:-2]
                            num_subtopic = len(original_A1)
                            temp = query[:-2]
                            temp_sum = sum(temp)
                            query =[temp[i]/temp_sum for i in range(len(temp)) ]
                        else:
                            original_A1 = dictionary_A1
                            num_subtopic=len(original_A1)
                            
                        train_subtopics = np.empty((len(subtopic_candidates), len(original_A1[0])))
                        train_subtopics_code = np.empty((len(subtopic_candidates), len(original_A1)))
                        subtopic_words ={}
                        for index,subtopic in enumerate(subtopic_candidates):
                            if flag == False:
                                train_subtopics[index]=subtopic.get_query_vec_tfidf()
                                train_subtopics_code[index]=subtopic.get_true_sprase_vec()[2:]
                            else:
                                train_subtopics[index]=subtopic.get_query_vec_tfidf()
                                train_subtopics_code[index]=subtopic.get_true_sprase_vec()
                            cid =  subtopic.get_query_label()
                            if subtopic_words.has_key(cid):
                                subtopic_words[cid].extend(subtopic.get_query_words())
                            else:
                                subtopic_words[cid]=[]
                                subtopic_words[cid].extend(subtopic.get_query_words())
                        dictionary = Dictionary(num_subtopic,train_subtopics,train_subtopics_code,original_A1,dictionary_loss,para_1)

                        doc_rep = []
                        for document in documents:
                            if dict_learning== "N1"or dict_learning=="N3":
                                sparse_coding = dictionary.document_sparse_representation_n1(document)
                            elif dict_learning == "N2" or dict_learning=="N4":
                                sparse_coding = dictionary.document_sparse_representation_n2(document)
                            elif dict_learning == "L1" or dict_learning == "L3":
                                sparse_coding = dictionary.document_sparse_representation_l13(document)
                            elif dict_learning == "L2" or dict_learning == "L4":
                                sparse_coding = dictionary.document_sparse_representation_l24(document)         
                            elif dict_learning == "L5" or dict_learning == "L7":
                                sparse_coding = dictionary.document_sparse_representation_l57(document)
                            elif dict_learning == "L6" or dict_learning == "L8":
                                sparse_coding = dictionary.document_sparse_representation_l68(document)
                            
                            if len(sparse_coding)==1:
                                document.set_sparse_rep(sparse_coding[0])
                            else:
                                document.set_sparse_rep(sparse_coding)
                            rep = document.get_sparse_rep()
                            
                            doc = Document()
                            doc.set_id(document.get_id())
                            doc.set_doc_str(document.get_doc_str())
                            doc.set_related_sid(document.get_related_sid())
                            if dict_learning == "N1" or dict_learning == "N2" or dict_learning == "L1" or dict_learning == "L2" or dict_learning == "L5" or dict_learning == "L6": 
                                doc.set_sparse_rep([rep[i] if rep[i]>0 else 0 for i in range(len(rep))])
                            else:
                                doc.set_sparse_rep(rep)
                            doc.set_term_vec(document.get_term_vec())
                            doc.set_tfidf_vec(document.get_tfidf_vec())
                            doc.set_true_rank(document.get_true_rank())
                            doc.set_pos_vec(document.get_pos_vec())
                            doc_rep.append(doc)
                        sparse_representation_data.append([runname,doc_rep,query])     
    return sparse_representation_data
if __name__=="__main__":
    print "start..."
    basepath ="D:/diversification/ntcir09/"
    topicID = "0001"
    sparse_parameters ={}
    sparse_parameters["mine_method"]=["1"]#1:standard subtopic 2:auto mining subtopic
    sparse_parameters["dictionary_loss"] =["omp"]#'lasso_cd']#, 'lars','lasso_lars', 'threshold','omp']
    sparse_parameters["dictionary_rep"] =["S1","E6"]#,"E6","S2","S1"] #"E1","E2","E3","E4",
    sparse_parameters["dictionary_norm"] =["N"]
    sparse_parameters["document_ori_rep"] =["D5"]#[,"D5","D3"]"D1","D2","D4"]
    sparse_parameters["document_rep_norm"] =["N"]
    sparse_parameters["para_1"] =[1.2]
    sparse_representation(basepath,topicID,sparse_parameters)
    print "end."