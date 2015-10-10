# -*-coding: utf-8 -*-

import multiprocessing
import os
import pickle
import random
import string
import sys

from numpy import array
from sklearn import datasets, cluster, metrics

from candidaterepresentation import getCandidatesRepresentations
from doc import getDocList
from invertedIndex import Index
from loadconfigfile import config_init
from phraserepresentation import PhraseRepresentation, VectorBlend
from precomputing import *
from querypreprocessing import getStopWordList, getQueryListNoTopic,\
    getTopicWordList
from wordrepresentation import WordRepresentation


RUN_NAME="CNU_LAB"

def dataPreparation(candidateReps, option="VSM"):
    truelabels = []
    pointrows = []
    dim = 0
    for candRep in candidateReps:
        truelabels.append(candRep.label)
        vec = None
        option = option.replace(" ","")
        if option == "EmbeddingVec":
            vec = candRep.GetEmbeddingVec()
        elif option == "VSMI":
            vec=[]
            vec.extend(candRep.GetVSM())
            vec.extend(candRep.GetInvertedDocVec())
        elif option == "VSM":
            vec=[]
            vec.extend(candRep.GetVSM())
        elif option == "InvertedDocVec":
            vec = candRep.GetInvertedDocVec()
        elif option == "CategoryVec":
            vec = candRep.GetCategoryVec()
        elif option == "VIE":
            vsm = candRep.GetVSM()
            idv = candRep.GetInvertedDocVec()
            emv = candRep.GetEmbeddingVec()
            # cat = candRep.GetCategoryVec()
            vsm.extend(idv)
            vsm.extend(emv)
            # vsm.extend(cat)
            vec = vsm
        elif option == "VIEC":
          
            vsm = list(candRep.GetVSM())
            idv = candRep.GetInvertedDocVec()
            emv = candRep.GetEmbeddingVec()
            cat = candRep.GetCategoryVec()
#             vsm.extend(idv)
            vsm.extend(emv)
            vsm.extend(cat)
            vec = vsm
            
        elif option == "EC":
            vec = []
            emv = candRep.GetEmbeddingVec()
            cat = candRep.GetCategoryVec()
            vec.extend(cat)
            vec.extend(emv)
        elif option == "IU":
            vec = []
            iuv = candRep.GetURLVec()
            vec.extend(iuv)
        elif option == "EW":
            vec = []
            ewv = candRep.GetEmbeddingWordVec()
            vec.extend(ewv)
        elif option == "EWE":
            vec = []
            ewv = candRep.GetEmbeddingWordVec()
            emv = candRep.GetEmbeddingVec()
            vec.extend(emv)
            vec.extend(ewv)
        if vec != None:
            pointrows.append(vec)
    return truelabels, array(pointrows)

def wordDataPreparation(wordreps, option="EmbeddingVec"):
    pointrows = []
    for word in wordreps:
        vec = None
        if option == "EmbeddingVec":
            vec = word.GetEmbeddingVec()
        if option == "EC":
            vec = []
            vec.extend(word.GetEmbeddingVec())
            vec.extend(word.GetCategoryVec())
        if vec != None:
            pointrows.append(vec)
    return pointrows

def DBScan(pointarrays, candforpre=None, eps=0.3, min_n=4):  # ,
    if candforpre == None:
        dis_matrix,ave = precomputing_distance(pointarrays)
        db = cluster.DBSCAN(eps=eps, min_samples=min_n, metric='precomputed').fit(dis_matrix)
        return db.labels_, db.core_sample_indices_,None
    else:
        dis_matrix = precomputing_distance(candforpre)
        dis_matrix_pre = precomputing_distance_pre(candforpre, pointarrays)
        db = cluster.DBSCAN(eps=eps, min_samples=min_n, metric='precomputed').fit(dis_matrix)
        lables = db.fit_predict(dis_matrix_pre)
        return lables, None,db.labels_
    
def affinitypropagation(pointarrays,candforpre=None, preference=None):
    ap = cluster.AffinityPropagation()
    if candforpre == None: 
        ap.fit(array(pointarrays))
        return ap.labels_, ap.cluster_centers_indices_,None
    else:
        ap.fit(array(candforpre))
        labels = ap.fit_predict(array(pointarrays))
        return labels,None,None

def kmeans(k=30, pointarrays=None, candforpre=None):
    k_means = cluster.KMeans(n_clusters=k, max_iter=100)
    if candforpre == None:
        k_means.fit(pointarrays) 
        cluster_indexes = k_means.predict(pointarrays)
        return cluster_indexes,None,None
    else:
        k_means.fit(candforpre)
        word_indexes = k_means.predict(candforpre)
        cluster_indexes = k_means.predict(pointarrays)
        return cluster_indexes,None,word_indexes

def getClusterResult(cluster_indexes):
    resultdict = {}
    for pid, cid in enumerate(cluster_indexes):
        if resultdict.has_key(cid):
            resultdict[cid].append(pid)
        else:
            resultdict[cid] = [pid]
    return resultdict

def printClusters(resultdict, candidateReps, o):
    for cid, points in resultdict.items():
        o.write("cluster " + str(cid) + "\n")
        elementStr = ""
        for  pid in points:
            if isinstance(candidateReps[pid], PhraseRepresentation):
                elementStr += candidateReps[pid].phrasestr + " | "
            elif isinstance(candidateReps[pid], list):
                cstr = "("
                for cand in candidateReps[pid]:
                    cstr += cand.phrasestr + " | "
                cstr += ")\n"
                elementStr += cstr
        o.write(elementStr)
        o.write("\n------------------------------------------------\n\n\n")

def printWordClusters(resultdict, candidateReps, o):
    for cid, points in resultdict.items():
        o.write("cluster " + str(cid) + "\n")
        elementStr = ""
        for  pid in points:
            if isinstance(candidateReps[pid], WordRepresentation):
                elementStr += candidateReps[pid].str + " | "
            elif isinstance(candidateReps[pid], list):
                cstr = "("
                for cand in candidateReps[pid]:
                    cstr += cand.str + " | "
                cstr += ")\n"
                elementStr += cstr
#         elementStr = elementStr.strip().replace(" ", " | ")
        o.write(elementStr)
        o.write('\n')
        o.write("------------------------------------------------")
        o.write('\n')
        o.write('\n')
        
    
def printClustersNTCIR2(filename,topicwordlist,cluster_indexes,cluster_candreps,second_centers,candidateReps,numbers,o):
    
    tid = filename.split(".")[0]
    cate_reps={}
    reps = {}
    num_reps = {}
    for index, cid in enumerate(cluster_indexes):
        if cate_reps.has_key(cid):
            cate_reps[cid].append(index)
            reps[cid].extend(cluster_candreps[index])
            num_reps[cid] += len(cluster_candreps[index])*1.0
        else:
            cate_reps[cid] = [index]
            reps[cid]=[] 
            reps[cid].extend(cluster_candreps[index])
            num_reps[cid] = len(cluster_candreps[index])*1.0
    lines = []
    #cid_h is the first cluster id, cids_l is the set of second cluster id
    for cid_h,cids_l in cate_reps.items():
        
        #使用该大类中出现次数最多的单词+原始查询作为概括
        worddict = {}
        for pid in cids_l:
            for cand in cluster_candreps[pid]:
                for word in cand.wordreps:
                    if len(word.str)<4:
                        continue
                    if worddict.has_key(word.str):
                        worddict[word.str]+=1
                    else :
                        worddict[word.str]=1
        wordlist = worddict.items()
        wordlist.sort(cmp=None, key=lambda s:s[1], reverse=True)
        tagstr = ""
        if len(wordlist) >0 and wordlist[0][1]>1 and wordlist[0][0] not in topicwordlist :
            tagstr= reps[cid_h][0].ori_query.replace(" ","")+wordlist[0][0]
        else:
            tagstr =reps[cid_h][0].phrasestr
        
        elementStr = tid+";0;"+ tagstr.strip() +';'+str(cid_h+1).split('.')[0]+";"+str(num_reps[cid_h]/numbers)[0:8]+";0;"
        for pid in cids_l:
            num = len(cluster_candreps[pid])
            c = str(num*1.0/numbers)[0:8]
            candidate = second_centers[pid]
            write_Str = elementStr+candidate.phrasestr+";"+str(pid+1).split('.')[0]+";"+c+";"+RUN_NAME
            lines.append(write_Str)

    #初始化排序过程 
    second_dict = {}
    for line in lines:
        query = line.split(";")
       
        if not second_dict.has_key((query[7],query[8])):
            second_dict[(query[7],query[8])] = []
            second_dict[(query[7],query[8])].append(query)
    keys = second_dict.keys()
    keys.sort(cmp=None,key = lambda asd:asd[1],reverse=True)
    lines = []
    j = 1
    for key in keys:
        query = second_dict[key][0]
        query[7]=str(j)
        j+=1
        lines.append(query)
    
    firstdict={}   
    for query in lines:
        
        query[9]="CNU_LAB_SYS5\n"
        if firstdict.has_key((query[3],query[4])):
            firstdict[(query[3],query[4])].append(query)
        else:
            firstdict[(query[3],query[4])]=[]
            firstdict[(query[3],query[4])].append(query)
    keys = firstdict.keys()
    keys.sort(cmp=None,key = lambda asd:asd[1],reverse=True)
    j =1
    for key in keys:
        
        queryes = firstdict[key]
        for query in queryes:
            query[3] =str(j)
            print ";".join(query)
            o.write(";".join(query))
        j+=1
        if j>5:break
     
def categorization(candreps):
    cate_weight = {}

    for cand in candreps:
        catvec = cand.GetCategoryVec()
        for cid, prob in enumerate(catvec):
            if cate_weight.has_key(cid):
                cate_weight[cid] += prob
            else:
                cate_weight[cid] = prob
    cate_weight_sorted = sorted(cate_weight.items(), key=lambda item: item[1], reverse=True)
    return cate_weight_sorted[0][0]


def distance(v1, v2):
    if len(v1) != len(v2):
        print "error"
        return
    sumv1 = sumv2 = sum1 = 0.0 
    for i in range(len(v1)):
        point = v1[i] - v2[i]
        sum1 += point * point
        sumv1 += v1[i] * v1[i]
        sumv2 += v2[i] * v2[i]
    if sumv2 > 0 and sumv1 > 0:
        sim = math.sqrt(sum1) / (math.sqrt(sumv1) * math.sqrt(sumv2))
    return sim
    
def Eu_distance(v1, v2):
    if len(v1) != len(v2):
        print "error"
        return
    sumv1 = sumv2 = sum1 = 0.0 
    for i in range(len(v1)):
        point = v1[i] - v2[i]
        sum1 += point * point
    sim = math.sqrt(sum1) 
    return sim

def buildDqrels():
    DICT_USE={}
    uuid = 1
    filein  = open("result_584")
    lines = filein.readlines()
    for line in lines:
        line = line.replace("\n","")
        items = line.split(";")
        if not DICT_USE.has_key(items[2]):
            DICT_USE[items[2]] = str(uuid).zfill(5)
            uuid +=1

    return DICT_USE
def subtopic_4diversification(cluster_dict,candidateReps):
    # subtopic_candidates,topic_words,query_dict
    # subtopic_candidates:[subtopic_str,label,terms,subtopic_frequency,urls]
    subtopic_candidates=[]
    query_dict={}
    subtopic_count = 0.0
  
    for cid, points in cluster_dict.items():
        for  pid in points:
            if isinstance(candidateReps[pid], PhraseRepresentation):
                cand = candidateReps[pid]
                subtopic_str = cand.phrasestr
                label = int(cid)
                terms =  cand.wordlist
                if terms.count(cand.ori_query)>0:
                    terms.remove(cand.ori_query)
                subtopic_frequency = cand.frequency
                urls = cand.url
                subtopic_candidates.append([subtopic_str,label,terms,float(subtopic_frequency),urls])
                if query_dict.has_key(cid):
                    query_dict[cid] += float(subtopic_frequency)
                else:
                    query_dict[cid] = float(subtopic_frequency)
                subtopic_count += float(subtopic_frequency)
    if subtopic_count == 0:
        subtopic_count = 1.0
    for key in query_dict.keys():
        query_dict[key] /= subtopic_count
        
    return  subtopic_candidates,query_dict       
             
           

def get_subtopic_clusters_4diversification(base_path,topic_ID):
    print topic_ID, "subtopic mining method"
    read_local_file="yes"
    second_cluster_method="KMeans"
    second_db_eps=0.3
    second_db_min_n=2
    second_ap_preference=0.5
    second_kmeans=15
    second_data_preparation="EmbeddingVec"
  
    pickle_file = base_path+"/pickle_mining/"+topic_ID
    if os.path.exists(pickle_file) and os.path.isfile(pickle_file):
        infile = open(pickle_file,"rb")
        subtopic_candidates = pickle.load(infile)
        query_dict  = pickle.load(infile)
        infile.close()
    else:
        queryfile = base_path+"subtopic/" + topic_ID
        docfilename = base_path + "doc_craw/" + topic_ID
        topicfilename = base_path + "topic/" + topic_ID
        
        if read_local_file == "yes":
            cdrs, wordReps, candembeddingdict = getCandidatesRepresentations(queryfile, topicfilename, docfilename,base_path,1)
        else:
            cdrs, wordReps, candembeddingdict = getCandidatesRepresentations(queryfile, topicfilename, docfilename,base_path,0)
        
        candidateReps = cdrs
        # 底层聚类 Second layer
        print len(candidateReps)
        labels_true, pointarrays = dataPreparation(candidateReps, option=second_data_preparation) 
        
        

        if second_cluster_method == "DBScan":
            labels_pred , centers ,unuselabel= DBScan(pointarrays, eps=second_db_eps,min_n=second_db_min_n)
        elif second_cluster_method == "AP":
            labels_pred , centers,unuselabel=affinitypropagation(pointarrays,preference=second_ap_preference)
        elif second_cluster_method == "KMeans":
            k_2= min(len(pointarrays),second_kmeans)
            labels_pred,centers,unuselabel =kmeans(k_2,pointarrays)  
        
        cluster_dict = getClusterResult(labels_pred)
        
        
        subtopic_candidates,query_dict= subtopic_4diversification(cluster_dict,cdrs)
        
        outfile = open(pickle_file,"wb")
        pickle.dump(subtopic_candidates,outfile) 
        pickle.dump(query_dict,outfile)        
        outfile.close()
        
        
    return subtopic_candidates,query_dict
        
def run_cl(basepath,topics):            
    for topic in topics:
        get_subtopic_clusters_4diversification(basepath,topic)         

if __name__ == '__main__':
    print""
    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topics = os.listdir(basepath+"rank/")
    topics.sort()
    topics1 = topics[96:]
    get_subtopic_clusters_4diversification(basepath,"0097") 
#     threads=[]
#     for i in range(10):
#         ts1 = topics1[i*10:i*10+10]
#                
#         t1=  multiprocessing.Process(target=run_cl,args=([basepath,ts1]))
#         threads.append(t1)
#             
#                
#     for i in range(10):
#         threads[i].start()
#     for i in range(10):
#         threads[i].join()