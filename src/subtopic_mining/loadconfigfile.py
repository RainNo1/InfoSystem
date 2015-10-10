# -*- coding: utf-8 -*-
def config_init(config_file):
    config={}
    config_set =set(["input_path" ,"result_path","read_local_file","word_cluster_outpath","first_cluster_method",
                 "first_expansion","first_db_eps","first_db_min_n","first_ap_preference","first_kmeans","second_cluster_method","second_expansion"
                ,"second_db_eps","second_db_min_n","second_ap_preference","second_kmeans","annotation","word_sim","stopword_file",
                "first_data_preparation","second_data_preparation","word_cluster_out"])
    config_num_set = set(["first_db_eps","first_db_min_n","first_ap_preference","first_kmeans","second_db_eps","second_db_min_n","second_ap_preference","second_kmeans","word_sim"])
    config_int_set = set(["first_kmeans","second_kmeans"])
    config["input_path"]=""
    config["result_path"]=""
    config["read_local_file"]="yes"
    config["word_cluster_outpath"]=""
    config["first_cluster_method"]="KMeans"
    config["first_expansion"]="no"
    config["word_cluster_out"]="no"
    config["first_db_eps"]=0.3
    config["first_db_min_n"]=2
    config["first_ap_preference"]=0.5
    config["first_kmeans"]=5
    config["second_cluster_method"]="KMeans"
    config["second_expansion"]="yes"
    config["second_db_eps"]=0.3
    config["second_db_min_n"]=2
    config["second_ap_preference"]=0.5
    config["second_kmeans"]=30
    config["annotation"]="readme.txt"
    config["word_sim"]=0.5
    config["stopword_file"]="/users/songwei/xuwenbin/subtopic/ntcir09/stoplist_utf8.txt"
    config["first_data_preparation"]="VSM"
    config["second_data_preparation"]="VSM"
    
    with open(config_file) as lines:
        for line in lines:
            line = line.replace("\n","")
            line = line.replace(" ","")
            if line.startswith("#"):
                continue
            item = line.split("=")
            if len(item)==2:
                key = item[0].strip() 
                if  key in config_set and item[1].strip()!="":
                    if key in config_num_set:
                        if key in config_int_set:
                            config[key] = int(item[1])
                        else:
                            config[key] = float(item[1])
                    else:
                        config[key] = item[1].strip()
    result_name = ""               
    if config["second_cluster_method"]=="KMeans":
        result_name+="K-"
        result_name+= str(config["second_kmeans"])
    elif config["second_cluster_method"]=="AP":
        result_name+="AP-"
        result_name+= str(config["second_ap_preference"])
    elif config["second_cluster_method"]=="DBScan":
        result_name+="DB-"
        result_name+= str(config["second_db_eps"])
    result_name+= "-"+config["second_data_preparation"]
    
    if  config["second_expansion"]=="yes":
        result_name+="-ex-"
    else:
        result_name+="-"
        
    if config["first_cluster_method"]=="KMeans":
        result_name+="K-"
        result_name+= str(config["first_kmeans"])
    elif config["first_cluster_method"]=="AP":
        result_name+="AP-"
        result_name+= str(config["first_ap_preference"])
    elif config["first_cluster_method"]=="DBScan":
        result_name+="DB-"
        result_name+= str(config["first_db_eps"])
    result_name+="-"+config["first_data_preparation"]
    
    if  config["first_expansion"]=="yes":
        result_name+="-ex"
    print result_name
    config["result_name"] = result_name
    return  config

if __name__ == "__main__":
    print "start"
    print min(32,12)
    print config_init("config")["first_data_preparation"]