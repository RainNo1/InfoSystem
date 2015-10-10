#-*- coding:utf8 -*-
import math
import multiprocessing
import os
import sys

sys.path.append("..")

from dictionary_learning.sparse_representation import sparse_representation
from result.result_analysis import DR_result_analysis, DR_result_analysis_sparse
from util.evaluation import call_eval_for_result
from util.runname import build_run_name, construct_doc_str
from util.vector_operation import vector_multiplier, vector_blend



def sparse_base_ranking(basepath,topicID,sparse_parameters,runlist):
    """
    build a document ranking list using document basic sparse representation method
    ------------
    parameters:
    ------------
        basepath: nctir09 or ntcir11 or ntcir10
        topicID: the topic ids
        sparse_parameters: a dictionary store the information as follow:
            dictionary_loss:dictionary Optimization Methods("lars","omp","lasso_lars","lasso_cd")
            dictionary_rep: the dictionary original representation(1-tf score addition,2-tf*term_weight )
            dictionary_norm: the dictionary original is normalization or not(Y-yes.N-not) 
            document_ori_rep:the document original representation(D1-tf/idf,)
            document_rep_norm: the document original representation is normalization or not(Y-yes,N-not)
        runlist: a file store the run name for the evaluation. 
    
    """
  
    print topicID,"ranking method start:"
    sparse_rep_all = sparse_representation(basepath,topicID,sparse_parameters)
    for runname1,documents,query in sparse_rep_all:
        for para_2 in sparse_parameters["para_2"]:
            for para_3 in sparse_parameters["para_3"]:
                runname =  runname1+str(int(para_2*10)).zfill(2)+str(int(para_3*10)).zfill(2)
                print topicID, "diversification ranking",runname
                sparse_base_ranking =[]
                document_selected=set()
                out_file = open(basepath+"rank/"+topicID+"/"+runname,"w")
                print >>runlist,runname
                w = None
                if topicID == "0001" or topicID == "0002" or topicID=="0201"or topicID=="0202":
                    if not os.path.exists("doctest/"+topicID+"/"):
                        os.mkdir("doctest/"+topicID+"/")
                    w = open("doctest/"+topicID+"/"+runname,"w") 
                # iterative get the best docs add to docs ranking list
                for i in range(len(documents)):
                    max_score,max_score_index = best_document_select(query,documents,sparse_base_ranking,document_selected,para_2,para_3,w)
                    best_document = documents[max_score_index]
                    best_document.set_ranking_score(max_score)
                    sparse_base_ranking.append(best_document)
                    document_selected.add(best_document.get_id())
                    if len(document_selected)>15:
                        break
                     
                # print the docs ranking list to file for evaluation
                for index,document in enumerate(sparse_base_ranking):
                    documentID = document.get_id()
                    document_rank = index+1
                    document_score = document.get_ranking_score()
                    print >>out_file,topicID +" Q0 "+documentID+" "+str(document_rank) +" "+str(document_score)+ " "+ runname
                out_file.close()


      
def best_document_select(query,documents,sparse_base_ranking,document_selected,para_2,para_3,w):
    ""
    max_score = -10000.0
    max_score_index = -1
    r = 0.0
    d = 0.0
    ea = 0.0
    eb = 0.0
    rank = ""
    rep = None
    document_id = ""
    for dindex,document in enumerate(documents):
        did = document.get_id()
        if did not in  document_selected:
            score ,relate_score, diversify_score,exponent_a,exponent_b= score_computing(document,sparse_base_ranking,query,para_2,para_3)
            if score>max_score:
                max_score = score
                max_score_index = dindex
                r = relate_score
                d = diversify_score
                ea = exponent_a
                eb = exponent_b
                rank = document.get_true_rank()
                rep=document.get_sparse_rep()
                re = document.get_related_sid()
                document_id = did
    if w is not None:
        print >>w,"%2.10f\t%2.10f\t%2.10f\t%2.10f\t%2.10f\t%3s\t%s\t%s"% (max_score,r,d,ea,eb,rank,"["+" ".join([str("%1.6f"%rep[i]) for i in range(len(rep))])+"]",str(re))
    return max_score,max_score_index
            

def score_computing(document,sparse_base_ranking,query,para_2,para_3):
    """
    """
    alpha = para_2
    vec_candidate = document.get_sparse_rep()
    relate_score = vector_multiplier(query,vec_candidate)
    
    if len(sparse_base_ranking)==0:
        return relate_score,0.0,0.0,0.0,0.0
    exponent_a = 0.0
    vec = [0.0 for i in range(len(vec_candidate))]
    for document_selected in sparse_base_ranking:
        vec_selected = document_selected.get_sparse_rep()
        vec = vector_blend(vec,vec_selected)
        exponent_a += vector_multiplier(vec_candidate,vec_selected)
    e_vec = [0.0 for i in range(len(vec_candidate))]
    for i in range(len(e_vec)):
        if vec[i]== 0.0:
            e_vec[i] = query[i]
    exponent_b = 0.5*vector_multiplier(vec_candidate,e_vec)
    diversify_score = (1.0/len(sparse_base_ranking))*(math.exp(-1*(exponent_a-para_3*exponent_b)))#/(len(sparse_base_ranking))
    
    return relate_score +alpha*diversify_score,relate_score, diversify_score,exponent_a,exponent_b
    
def run_ranking(basepath,topics,sparse_parameters):    
    
    for topicID in topics:
        if topicID == "10001":continue
        runlist = open(basepath+"rank/"+topicID+"/runlist","w")
        sparse_base_ranking(basepath,topicID,sparse_parameters,runlist)
        runlist.close()
        r = call_eval_for_result(basepath,topicID,10) 

def run_analysis(sparse_parameters,basepath,collection):
    runname_list = []
    for dictionary_loss in sparse_parameters["dictionary_loss"]:
        for dictionary_norm in sparse_parameters["dictionary_norm"]:
            for dictionary_rep in sparse_parameters["dictionary_rep"]:
                for document_ori_rep in sparse_parameters["document_ori_rep"]:
                    for document_rep_norm in sparse_parameters["document_rep_norm"]:
                        for mine_method in sparse_parameters["mine_method"]:
                            runname = build_run_name(dictionary_loss,dictionary_norm,dictionary_rep,document_ori_rep,document_rep_norm,mine_method)
                            for para_1 in sparse_parameters["para_1"]:
                                for subtopic_less in sparse_parameters["subtopic_less"]:
                                    for dict_learning in sparse_parameters["dictionary_learning"]:
                                        for para_2 in sparse_parameters["para_2"]:
                                            for para_3 in sparse_parameters["para_3"]:
                                                if subtopic_less=="Y":
                                                    runname_list.append(runname+str(int(para_1*10)).zfill(2)+"LY"+dict_learning+str(int(para_2*10)).zfill(2)+str(int(para_3*10)).zfill(2))
                                                else:
                                                    runname_list.append(runname+str(int(para_1*10)).zfill(2)+"LN"+dict_learning+str(int(para_2*10)).zfill(2)+str(int(para_3*10)).zfill(2))
    for runname in runname_list:
        doc_str = construct_doc_str(runname)
        DR_result_analysis_sparse(basepath,doc_str,runname,collection) 
        
        
def sysrun( topics1,basepath,sparse_parameters):
    threads=[]
    for i in range(10): 
        ts1 = [topics1[j] if int(topics1[j])%10== i else "10001" for j in range(len(topics1))]   
        t1=  multiprocessing.Process(target=run_ranking,args=([basepath,ts1,sparse_parameters]))
        threads.append(t1)       
    for i in range(10):
        threads[i].start()
    for i in range(10):
        threads[i].join()
    run_analysis(sparse_parameters,basepath,basepath.split("/")[-2])   
if __name__=="__main__":
    print "start..."

    # system parameters
    sparse_parameters ={}
    sparse_parameters["mine_method"]=["1","2"]#1:standard subtopic 2:auto mining subtopic
    sparse_parameters["subtopic_less"]=["Y","N"]# N:all subtopics we got,Y: less some subtopics
    sparse_parameters["dictionary_loss"] =["lasso_cd"]#'lasso_cd']#, 'lars', 'threshold',]
    sparse_parameters["dictionary_rep"] =["S1","E5"]#,"E1","E2"] #"E1","E2","E3","E4",["S2","S1","E6",
    sparse_parameters["dictionary_norm"] =["N"]
    sparse_parameters["dictionary_learning"]=["N1","N3","L2"]
    sparse_parameters["document_ori_rep"] =["D1"]#[,"D5","D3"]"D1","D2","D4"]
    sparse_parameters["document_rep_norm"] =["N"]
    sparse_parameters["para_1"] =[0.9,0.7]#[0.1*i+0.5 for i in range(7)]#[0.6,1.2]#
    sparse_parameters["para_2"] =[0.5]#[0.1*i for i in range(10)[0::2]]#
    sparse_parameters["para_3"] = [0.9]#[0.1*i for i in range(20)][7:14]#
    

    
#     run_ranking(basepath,["0201","0220"],sparse_parameters)
#     raise
    basepath="/users/songwei/xuwenbin/diversification/ntcir10/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
    sysrun(topics, basepath, sparse_parameters)
    basepath="/users/songwei/xuwenbin/diversification/ntcir09/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
    sysrun(topics, basepath, sparse_parameters)

    print "Done."
    
    