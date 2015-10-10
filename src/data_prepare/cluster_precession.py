#-*- coding:utf8 -*-
from numpy import array
from sklearn import cluster

from util.precomputing import precomputing_distance, precomputing_distance_pre


def DBScan(words, querys=None, eps=0.3, min_n=1):  # ,
    if querys == None:
        dis_matrix= precomputing_distance(words)
        db = cluster.DBSCAN(eps=eps, min_samples=min_n, metric='precomputed').fit(dis_matrix)
        return db.labels_, db.core_sample_indices_,None
    else:
  
        dis_matrix = precomputing_distance(words)
        print dis_matrix
        dis_matrix_pre = precomputing_distance_pre(words, querys)
        db = cluster.DBSCAN(eps=eps, min_samples=min_n, metric='precomputed').fit(dis_matrix)
        labels = [1 for i in range(len(querys))]
        for index,query in enumerate(querys):
            
            max_sim = -1000
            max_cluster_id = 0
            print db.core_sample_indices_
            print db.labels_
            for center_index in db.core_sample_indices_:
               
                if max_sim < dis_matrix_pre[len(words)+index][center_index]:
                    max_sim = dis_matrix_pre[len(words)+index][center_index]
                    max_cluster_id = db.labels_[center_index]
            labels[index] = max_cluster_id
        
        return db.labels_, None,labels 
    
def affinitypropagation(words,querys=None, preference=None):
    ap = cluster.AffinityPropagation(0.6)
    if querys == None: 
        ap.fit(array(words))
        return ap.labels_, ap.cluster_centers_indices_,None
    else:
        ap.fit(array(words))
        w_labels = ap.labels_
        labels = ap.fit_predict(array(querys))
        return w_labels,None,labels

def kmeans(k=30, words=None, querys=None):
    k_means = cluster.KMeans(n_clusters=k, max_iter=100)
    if querys == None:
        k_means.fit(words) 
        cluster_indexes = k_means.predict(words)
        return cluster_indexes,None,None
    else:
        k_means.fit(words)
        query_indexes = k_means.predict(words)
        cluster_indexes = k_means.predict(querys)
        return query_indexes,None,cluster_indexes
    

   
def cluster_engine(words,querys = None,cluster_model="DBScan"):
    """
    cluster the expansion words,using AP or DBScan.
    -----------
    parameters:
    -----------
        words: prepared word for clustering.
        weight_model: the cluster weight computing function.(term number ratio, distance ratio of the original query,)
        word_detail: a dictionary store the representation information for word.  
        cluster_model: the cluster algorithm,include(DBScan or AP)
    -----------
    return:
    -----------
        a dictionary of clusters.
        a weight dictionary of each cluster.
        a vector of clusters weight, the weight as the relevance score of the subtopic of original query.
    """
    cluster_dict = {}
    query_dict={}
    flag = True
    db = 0.3
    if querys != None:
        if cluster_model== "KMeans":
            labels_word , centers ,query_label=  kmeans(20,words, querys)
            
    else:
        if cluster_model =="DBScan":
            # labels_pred:the predict label,centers:the center words of each cluster
            labels_word , centers ,query_label=  DBScan(words, eps=0.40, min_n=4)
    
    
    # get the dictionary of cluster cluster_id:[word_index]
    for index, cid in enumerate(labels_word):
        
        cluster_id = int(cid)
     
        if cluster_dict.has_key(cluster_id):
            cluster_dict[cluster_id].append(index)
        else:
            cluster_dict[cluster_id] = [index]
    if querys != None:
        for index,qid in enumerate(query_label):
            if query_dict.has_key(qid):
                query_dict[qid].append(index)
            else:
                query_dict[qid] = [index]
        
    return cluster_dict,query_dict
        

if __name__=="__main__":
    print "start..."
    pointarrays = [[0,0,0,1,1,0],[0,0,0,0,2,1],[0,0,0,1,2,0],[0,0,0,1,2,0]]
    cand =  [[1,0,0,5,4,0],[5,4,7,3,1,5]]
    print DBScan( pointarrays,cand)