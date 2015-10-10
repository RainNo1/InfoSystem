#-*- coding:utf8 -*-
import os
import pickle
import sys
# sys.path.append("..")
from base_model.term_expansion import WordEmbeddingFramework
from data_prepare.cluster_precession import cluster_engine
from subtopic_mining.clusteringengine import get_subtopic_clusters_4diversification
from util.isgoodword import isGoodWord
from util.vector_operation import vectors_blend


# Extract query reformulation into a list
def get_subtopic_candidate(basepath,topicID):
    """
    get the subtopic of the original query. return a list with subtopic information.
    for each data_prepare,would build a word dict to store the information of terms.
    ------------
    return:
    ------------
    subtopic_list: a list with [id,label,words,URLs]
    url set: url set.
    word2id_weight: word id and word appeared subtopic
    subtopic_length_average,subtopic_term_number_all,topic_word_set,query
    [ID,label,words,frequency,urls]
    
    """
    subtopic_path = basepath + "subtopic/"
    topic_path = basepath+"topic/"
    f = open(subtopic_path+topicID)
    lines = f.readlines()
    f.close()
    query_probability_dict = get_orinal_query_vec(basepath,topicID)
    subtopic_term_number_all = 0.0
    subtopic_number = 0
    subtopic_length_sum = 0.0
    word2id_weight = {}
    subtopic_list = []   
    urlset = set()
    wid = 0
    
    topic_word_set = get_topic_words(topic_path+topicID)
    for line in lines:
        elements = line.strip().split('||')
        if len(elements)!=4:continue
        label = int(elements[0])
        words = elements[1].split("\t")
        subtopic_frequency = int(elements[2])
        if len(elements[3])>1:
            urls = elements[3].split("\t")
        else:
            urls = [elements[3].strip()]
        #processing the URL related 
        for url in urls:
            if url == 0:continue
            if url not in urlset:
                urlset.add(url)
        terms = []
        
        subtopicID = "".join([word.split("/")[0] for word in words ])
       
                
        for word in words:
            if  (not isGoodWord(word)) or word in topic_word_set :
                continue
            else:
                
                term,pos = word.split("/")
                terms.append(term)
                if word2id_weight.has_key(term):
                    word2id_weight[term][1] += subtopic_frequency
                    subtopic_term_number_all += subtopic_frequency
                    word2id_weight[term][2].append([subtopicID,subtopic_frequency])
                else:
                    subtopic_term_number_all += subtopic_frequency
                    word2id_weight[term]=[wid,subtopic_frequency,[[subtopicID,subtopic_frequency]]]
                    wid += 1
        if len(terms) == 0:
            continue
        subtopic_length_sum += subtopic_frequency * len(terms)
        subtopic_number += subtopic_frequency
        
        subtopic_list.append((subtopicID,label, terms,subtopic_frequency,urls))
    subtopic_length_average = subtopic_length_sum/subtopic_number

    return subtopic_list,urlset,word2id_weight,subtopic_length_average,subtopic_term_number_all,topic_word_set,query_probability_dict,subtopic_number


# Extract query reformulation into a list

def get_subtopic_candidate_for_baseline_mining(basepath,topicID):
    
    subtopic_candidates,topic_words,query_dict = read_subtopic_from_mining(basepath,topicID)
    word2id_weight = {}
    word_id = 0
    subtopic_length_average = 0.0
    subtopic_number = 0.0 
    subtopic_term_number_all = 0.0
    
    for subtopic_str,label,terms,subtopic_frequency,urls in subtopic_candidates:
        subtopic_term_number_all += len(terms)*subtopic_frequency
        subtopic_number  += subtopic_frequency
        for term in terms:
            if word2id_weight.has_key(term):
                word2id_weight[term][1] += subtopic_frequency
                word2id_weight[term][2].append([subtopic_str,subtopic_frequency])
            else:
                word2id_weight[term] =[]
                word2id_weight[term].append(word_id)
                word_id += 1
                word2id_weight[term].append(subtopic_frequency)
                word2id_weight[term].append([[subtopic_str,subtopic_frequency]])
    subtopic_length_average = subtopic_term_number_all/subtopic_number
    return subtopic_candidates,None,word2id_weight,subtopic_length_average,subtopic_term_number_all,topic_words,query_dict,subtopic_number

def read_subtopic_from_files(basepath,topicID):
    """
    get the subtopic of the original query. return a list with subtopic information.
    for each data_prepare,would build a word dict to store the information of terms.
    ------------
    return:
    ------------
    subtopic_list: a list with [id,label,words,URLs]
    subtopic_length_average,subtopic_term_number_all,topic_word_set,query
    [ID,label,words,frequency,urls]
    
    """
    subtopic_path = basepath + "subtopic/"
    topic_path = basepath+"topic/"
    f = open(subtopic_path+topicID)
    lines = f.readlines()
    f.close()
    query_probability_dict = get_orinal_query_vec(basepath,topicID)
    subtopic_list = []   
    urlset = set()
    
    topic_word_set = get_topic_words(topic_path+topicID)
    for line in lines:
        elements = line.strip().split('||')
        if len(elements)!=4:continue
        label = int(elements[0])
        words = elements[1].split("\t")
        subtopic_frequency = int(elements[2])
        if len(elements[3])>1:
            urls = elements[3].split("\t")
        else:
            urls = [elements[3].strip()]
        #processing the URL related 
        for url in urls:
            if url == 0:continue
            if url not in urlset:
                urlset.add(url)
        terms = []
        subtopicID = "".join([word.split("/")[0] for word in words ])
        for word in words:
            if  (not isGoodWord(word)) or word in topic_word_set :
                continue
            else:
                term,pos = word.split("/")
                terms.append(term)
                
        if len(terms) == 0:
            continue
      
        subtopic_list.append((subtopicID,label, terms,subtopic_frequency,urls))
 
    return subtopic_list,topic_word_set,query_probability_dict

def get_orinal_query_vec(basepath,topicID):
    files = basepath+"rank/"+topicID+'/standard.Iprob'
    query_dict={}
    with open(files) as lines:
        for line in lines:
            line = line.replace("\n","")
            item = line.split(" ")
            if len(item)==3:
                if not query_dict.has_key(int(item[1])):
                    query_dict[int(item[1])] = float(item[2])

    return query_dict

def get_topic_words(filename):
    topicwords = set()
    with open(filename)  as lines:
        for line in lines:
            for word in line.split("\t"):
                try:
                    term,pos = word.split("/")
                except:
                    continue
                if not word in topicwords:
                    topicwords.add(word)
    return topicwords
def read_subtopic_from_mining(basepath,topicID):
    subtopic_candidates,query_dict = get_subtopic_clusters_4diversification(basepath,topicID)
    topic_file_path = basepath+"topic/"
    topic_words = get_topic_words(topic_file_path+topicID)
    return subtopic_candidates,topic_words,query_dict

def subtopic_expansion(basepath,topicID,expansion_method,mining_method):
    """
    this function provides prepared information of subtopics.
    """
    print topicID,"subtopic expansion ",expansion_method,mining_method
    if not os.path.exists(basepath+"/pickle_expansion/"+topicID+"-"+mining_method+"/"):
        os.mkdir(basepath+"/pickle_expansion/"+topicID+"-"+mining_method+"/")
    pickle_file = basepath+"/pickle_expansion/"+topicID+"-"+mining_method+"/"+expansion_method
    if os.path.exists(pickle_file) and os.path.isfile(pickle_file):
        infile = open(pickle_file,"rb")
        subtopic_list = pickle.load(infile)
        word_details  = pickle.load(infile)
        topic_words = pickle.load(infile)
        query_dict = pickle.load(infile)
        infile.close()
    else:
        if mining_method == "1":
            subtopic_candidates,topic_words,query_dict = read_subtopic_from_files(basepath,topicID)
        else:
            subtopic_candidates,topic_words,query_dict = read_subtopic_from_mining(basepath,topicID)
        word_embedding = WordEmbeddingFramework()  
        subtopic_list = [] # (subtopicID,label,terms,subtopic_frequency,urls)
        word_details = {}  # {word:[embedding vector,frequency]}
        for subtopic_str,label,terms,subtopic_frequency,urls in subtopic_candidates:
            term_for_expansion = []
            subtopic_words =[]
            subtopic = []
            if expansion_method.startswith("E"):
                # E2,E6 and E4 expansion method require the Q+A for expansion
                if expansion_method == "E2" or expansion_method == "E4" or expansion_method == "E6":
                    for word in topic_words:
                        vec = word_embedding.get_embedding_vector(word)
                        if vec != None:
                            term_for_expansion.append(word)
                for word in terms:
                    vec = word_embedding.get_embedding_vector(word)
                    if vec == None:
                        continue
                    if not word_details.has_key(word):
                        word_details[word]=[vec]
                        word_details[word].append(subtopic_frequency)
                    term_for_expansion.append(word)
                    subtopic_words.append(word)
                    
                if len(term_for_expansion)==0:
                    print subtopic_str
                    continue
                sim_word_weight = word_embedding.get_expansion_words(term_for_expansion)
                if sim_word_weight == None: continue
                i =0
                if expansion_method =="E2" or expansion_method=="E1":
                    subtopic_words=[]
                for s_word,weight in sim_word_weight:
                    if not isGoodWord(s_word+"/n"):continue
                    i +=1
                    
                    if weight<0.6:continue
                    if word_details.has_key(s_word):
                        word_details[s_word][1]+= subtopic_frequency
                    else:
                        vec = word_embedding.get_embedding_vector(s_word)
                        if vec == None:continue
                        word_details[s_word]=[]
                        word_details[s_word].append(vec)
                        word_details[s_word].append(subtopic_frequency)
                    # E5 and E6 method use the expansion words and the original words as the subtopic words
                    if expansion_method == "E5" or expansion_method == "E6":
                        subtopic_words.append(s_word)
                        if i>3:break
                    if expansion_method == "E1" or expansion_method == "E2":
                        subtopic_words.append(s_word)
                        if i>5:break
            else:
                for word in terms:
                    vec = word_embedding.get_embedding_vector(word)
                    if vec == None:
                        continue
                    if not word_details.has_key(word):
                        word_details[word]=[vec]
                        word_details[word].append(subtopic_frequency)

                    subtopic_words.append(word)
               
            subtopic.append(subtopic_str)
            subtopic.append(label)
            subtopic.append(subtopic_words)
            subtopic.append(subtopic_frequency)
            subtopic.append(urls)
            subtopic_list.append(subtopic)
        outfile = open(pickle_file,"wb")
        pickle.dump(subtopic_list,outfile) 
        pickle.dump(word_details,outfile)
        pickle.dump(topic_words,outfile)
        pickle.dump(query_dict,outfile)
        outfile.close()
    
    return subtopic_list,word_details,topic_words,query_dict
                    
def prepare_subtopic(basepath,topicID,expansion_methods,mining_method):
    

    prepare_subtopic_dict = {}
    for expansion_method in expansion_methods:
        
        if prepare_subtopic_dict.has_key(expansion_method):
            continue        
        print topicID ,"preparing the subtopic:"+expansion_method+"-"+mining_method
        
        if not os.path.exists(basepath+"pickle_subtopic/"+topicID+"-"+mining_method+"/"):
            os.mkdir(basepath+"pickle_subtopic/"+topicID+"-"+mining_method+"/")
        pickle_file = basepath+"pickle_subtopic/"+topicID+"-"+mining_method+"/"+expansion_method
        if os.path.exists(pickle_file) and os.path.isfile(pickle_file):
            infile = open(pickle_file,"rb")
            prepare_subtopic = pickle.load(infile)
            prepare_subtopic_dict[expansion_method] = []
            prepare_subtopic_dict[expansion_method].extend(prepare_subtopic)
            infile.close()
        else:
            word2id_weight = {} #{word: [wid,word_frequency,[subtopicID,frequency]]}
            subtopic_dict={} # a list store the information of subtopics
            subtopic_lenght_ave = 0.0
            subtopic_term_number_all = 0.0
            subtopic_number = 0.0
            wid = 0
            query_dict={}
            
            subtopics,word_details,topic_words,query =subtopic_expansion(basepath,topicID,expansion_method,mining_method)
    #         ori_subtopic_word_num = 0.0
    #         ori_subtopic_num = 0.0
    #         for subtopicID,labels,words,subtopic_frequency,urls in subtopics:
    #             ori_subtopic_num += subtopic_frequency
    #             ori_subtopic_word_num += subtopic_frequency*len(words)
    
            if expansion_method.startswith("S") or expansion_method=="E5" or expansion_method=="E6" or expansion_method=="E1" or expansion_method=="E2":
                word_rep = []
                useful_word = {}
                words = word_details.keys()
             
                for word in words:
                    word_rep.append(word_details[word][0])
                cluster_dict,query_cluster = cluster_engine(word_rep,cluster_model="DBScan")
                cluster_ids = cluster_dict.keys()
#                 print cluster_ids
                for cid in cluster_ids:
                    if cid == -1.0:
#                         print cid
#                         print  "\t".join([words[i] for i in cluster_dict[cid]])
                        continue
                    else:
#                         print cid,
#                         print " \t".join([words[i] for i in cluster_dict[cid]])
                        for word in [words[i] for i in cluster_dict[cid]]:
                            if useful_word.has_key(word):
                                useful_word[word] +=1
                            else:
                                useful_word[word] =1
                for subtopicID,label, words,subtopic_frequency,urls in subtopics:
                    terms = []
                    for word in words:
                        if not useful_word.has_key(word):continue
                        terms.append(word)
                        if word2id_weight.has_key(word):
                            word2id_weight[word][1] += subtopic_frequency
                            subtopic_term_number_all += subtopic_frequency
                            word2id_weight[word][2].append([subtopicID,subtopic_frequency])
                        else:
                            subtopic_term_number_all += subtopic_frequency
                            word2id_weight[word]=[wid,subtopic_frequency,[[subtopicID,subtopic_frequency]]]
                            wid += 1
                    subtopic_number += subtopic_frequency
                    if len(terms)==0:
                        for word in words:
                            terms.append(word)
                            if word2id_weight.has_key(word):
                                word2id_weight[word][1] += subtopic_frequency
                                subtopic_term_number_all += subtopic_frequency
                                word2id_weight[word][2].append([subtopicID,subtopic_frequency])
                            else:
                                subtopic_term_number_all += subtopic_frequency
                                word2id_weight[word]=[wid,subtopic_frequency,[[subtopicID,subtopic_frequency]]]
                                wid += 1
                    if subtopic_dict.has_key(label):
                        subtopic_dict[label].append([subtopicID,label, terms,subtopic_frequency,urls])
                    else:
                        subtopic_dict[label]=[]
                        subtopic_dict[label].append([subtopicID,label, terms,subtopic_frequency,urls])
                query_dict = query
    #         elif expansion_method =="E1" or expansion_method=="E2":
    #             """ using the cluster of expansion words as subtopics"""
    #             word_rep = []
    #             words = word_details.keys()
    #             for word in words:
    #                 word_rep.append(word_details[word][0])
    #             cluster_dict,query_cluster = cluster_engine(word_rep,cluster_model="DBScan")
    #             cluster_ids = cluster_dict.keys()
    #             for cid in cluster_ids:
    #                 s_words = [words[i] for i in cluster_dict[cid]]
    #                 subtopicID = " ".join(s_words)
    #                 terms = []
    #                 for word in s_words:
    #                     terms.extend([word for i in range(int(word_details[word][1]))])
    #                     if word2id_weight.has_key(word):
    #                         word2id_weight[word][1] += word_details[word][1]
    #                         subtopic_term_number_all += word_details[word][1]
    #                         word2id_weight[word][2].append([subtopicID,1])
    #                     else:
    #                         subtopic_term_number_all += word_details[word][1]
    #                         word2id_weight[word]=[wid,word_details[word][1],[[subtopicID,1]]]
    #                         wid += 1
    #                 subtopic_number += 1
    #                 subtopic = [subtopicID,cid,terms,1,None]
    #                 if subtopic_dict.has_key(cid):
    #                     subtopic_dict[cid].append(subtopic)
    #                 else:
    #                     subtopic_dict[cid]=[subtopic]
    #                 query_dict[cid] = len(terms)*1.0
    #             for cid in query_dict.keys():
    #                 query_dict[cid] /= (subtopic_term_number_all+1)*1.0
            elif expansion_method=="E3" or expansion_method=="E4":
                # prepare the word for cluster
                word_rep = []
                subtopic_rep = []
                words = word_details.keys()
                for word in words:
                    word_rep.append(word_details[word][0])
                # prepare the subtopics for cluster
                for subtopicID,label, terms,subtopic_frequency,urls in subtopics:
                    subtopic_word_rep = []
                    subtopic_word_weight =[]
                    number = 0.0
                    for word in terms:
                        number+= word_details[word][1]
                        subtopic_word_rep.append(word_details[word][0])
                        subtopic_word_weight.append(word_details[word][1])
                    weight = [subtopic_word_weight[i]/number for i in range(len(subtopic_word_weight))]
                    vec = vectors_blend(subtopic_word_rep,weight)
                    subtopic_rep.append(vec)
        
                cluster_dict,query_cluster = cluster_engine(word_rep, subtopic_rep, "KMeans")
                cluster_ids = query_cluster.keys()
                for cid in cluster_ids:
                    s_words = [words[i] for i in cluster_dict[cid]]
                    subtopicID = " ".join(s_words)
                    terms = []
                    for word in s_words:
                        terms.extend([word for i in range(int(word_details[word][1]))])
                        if word2id_weight.has_key(word):
                            word2id_weight[word][1] += word_details[word][1]
                            subtopic_term_number_all += word_details[word][1]
                            word2id_weight[word][2].append([subtopicID,1])
                        else:
                            subtopic_term_number_all += word_details[word][1]
                            word2id_weight[word]=[wid,word_details[word][1],[[subtopicID,1]]]
                            wid += 1
                    subtopic_number += 1
                    subtopic = [subtopicID,cid,terms,1,None]
                    if subtopic_dict.has_key(cid):
                        subtopic_dict[cid].append(subtopic)
                    else:
                        subtopic_dict[cid] = [subtopic]
                    query_dict[cid] = len(terms)*1.0
                    
                for cid in query_dict.keys():
                    query_dict[cid] /= (subtopic_term_number_all+1)*1.0
    #                 if expansion_method =="E3":
    #                     query_dict[cid] /= (subtopic_term_number_all+1)*1.0
    #                 else:
    #                     query_dict[cid] = len(query_cluster[cid])/(1.0*sum([len(query_cluster[i]) for i in query_cluster.keys()]))
           
            if subtopic_number == 0:
                subtopic_number = 1.0
            subtopic_lenght_ave = 1.0*subtopic_term_number_all/(subtopic_number )
            
            prepare_subtopic = [subtopic_dict,word2id_weight,query_dict,subtopic_lenght_ave,topic_words,subtopics,subtopic_number]
            outfile = open(pickle_file,"wb")
            pickle.dump(prepare_subtopic,outfile) 
            outfile.close()
            
            prepare_subtopic_dict[expansion_method] = [subtopic_dict,word2id_weight,query_dict,subtopic_lenght_ave,topic_words,subtopics,subtopic_number]
    return  prepare_subtopic_dict
            
    
if __name__=="__main__":
    print "start..."
    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topicID = "0001"
    expansion_methods = ["E6","E5"]
    prepare_subtopic(basepath,topicID,expansion_methods,mining_method="1")

    
    
    
    
    
    
    
    
    
    
    