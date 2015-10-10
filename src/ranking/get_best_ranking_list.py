#-*- coding:utf8 -*-
import commands
import multiprocessing
import os


def read_doc_list(filename):
    """
        get the original docs list from docs list file.
        filename is the data_prepare index.
        return doc_dict: documents name and relevant subtopic index.
    """
    doc_dict = {}
    with open(filename) as lines:
        for line in lines:
            line = line.replace("\n","")
            items = line.split(" ")
            if doc_dict.has_key(items[2]):
                doc_dict[items[2]].append([items[1],items[3]])
            else:
                doc_dict[items[2]]=[]
                doc_dict[items[2]].append([items[1],items[3]])
    return doc_dict
    

def call_eval_for_best(inpath,topicID,cutoff):
    """
        for each file get the best run name in the run list.
        parameter:inpath,topicID,cutoff
        return: the best docs name
    """
    
    #change the work directory and save the original work directory
    ori_directory = os.getcwd()

    
    if not ori_directory == inpath:
        os.chdir(inpath)
    
    shell_command = ""
    dqrels_file =  "standard.Dqrels"
    iprob_file =  "standard.Iprob"
    et = str(int(topicID)%5)
    #detecting whether the first time call the evaluation function
    if cutoff == 1: 
        # built evaluation workspace
        #create Irelv , Grelv and din files
        shell_command = "/users/songwei/xuwenbin/eval"+et+"/DIN-splitqrels "+iprob_file+" "+dqrels_file+" test"
        commands.getstatusoutput(shell_command)
    
    #create res files
    shell_command_res = "cat ./runlist | /users/songwei/xuwenbin/eval"+et+"/TRECsplitruns ./standard.Iprob.tid"
    commands.getstatusoutput(shell_command_res)
    print "evaluation start."
    print "cat ./runlist | /users/songwei/xuwenbin/eval"+et+"/D-NTCIR-eval standard.Iprob.tid test "+str(cutoff)+" 110"
    #evaluate
    shell_command_eval = "cat ./runlist | /users/songwei/xuwenbin/eval"+et+"/D-NTCIR-eval standard.Iprob.tid test "+str(cutoff)+" 110"
    item = commands.getstatusoutput(shell_command_eval)
#     if cutoff == 1:
#         print item
    
    #get the best docs
    nDCG= 0.0
    best_doc = ""
    file_res = os.listdir(inpath)
    for res in file_res:
        items = res.split(".")
        if len(items)==3 and items[-1]=="Dnev":
             
            o = open(inpath+res)
            lines = o.readlines()
            o.close()
            for line in lines:
                temps = line.split(" ")
                
                if temps[0]==topicID and temps[1].lstrip().startswith("D#-nDCG@") and float(temps[-1].replace("\n",""))<1.0 and float(temps[-1].replace("\n",""))>nDCG:
                    best_doc = items[0]
                    nDCG  = float(temps[-1].replace("\n",""))
        else:
            continue
        
    
    #reducing the original work directory
    os.chdir(ori_directory)
    return best_doc
def constructor(documents):
    ""
    rank_list = []
    if len(documents) == 1:
        return documents
    else:
        for document in documents:
            docs = []
            docs.extend(documents)
            docs.remove(document)
            for item in constructor(docs):
                temp = [document]
                temp.extend(item)
                rank_list.append(temp)
    return rank_list
       
def construct_rank_list(rank_file_path,topicID):
    """
       iterator construct the ideal ranking list as training data.
       parameter: doc_dict,documents name list without docs content.
       return: Null.
       output: file-rank_list,the ideal docs name list.
    """
    filename = rank_file_path+topicID
    doc_dict = read_doc_list(filename+"/standard.Dqrels")
    out_path = rank_file_path+topicID+"/"
    rank_doc_list=[]
    ori_doc_set = set(doc_dict.keys())
    num = 1
    doc_list_len = len(ori_doc_set)
    

    while num < doc_list_len:
        rank_list = []
        for doc_name in ori_doc_set:
            temp = []
            temp.extend(rank_doc_list)
            temp.append(doc_name)
            rank_list.append(temp)
        print topicID,"select %d document"%num
        #print the runlist to files
        runlist = open(out_path+"runlist","w")
        for items in rank_list:
            out_name =  items[-1]
            if items == "":
                continue
            file_out = open(out_path+out_name,"w")
            print>>runlist,out_name
            for item in items:      # iterator write the run_result to file and named as the last docs
                print>>file_out, topicID+" Q0 "+item+" "+str(num)+" "+str(doc_list_len-num+1)+" TESTRUN"
            file_out.close()
        runlist.close()
        
        
        #call the evaluation function to find the best docs 
        best_doc = call_eval_for_best(out_path,topicID,num)
        ori_doc_set.remove(best_doc)
        rank_doc_list.append(best_doc)
        num += 1
        
        #delete the useless files
        files = os.listdir(out_path)
        file_filter = set()
        file_filter.add(topicID)
        file_filter.add("standard.Dqrels")
        file_filter.add("standard.Iprob")
        file_filter.add("standard.Iprob.tid")
        file_filter.add("ideal.rank")
        
        for name in files:
            if name not in file_filter and not name.lstrip().startswith(".nfs"):
                os.remove(out_path+name)
        files = os.listdir(out_path+topicID+"/")
        for name in files:
            if name.lstrip().startswith(".nfs"):continue
            if name.split(".")[-1]=="res" or name.split(".")[-1]=="glab":
                os.remove(out_path+topicID+"/"+name)
       
    # output the ideal ranking list to file
    result_file = open(out_path + "ideal.rank","w")
    rank_index= 1
    for doc in rank_doc_list:
        print >> result_file,topicID+" Q0 "+doc+" "+str(rank_index)+" "+str(1000.0-rank_index)+" TESTRUN"
        rank_index += 1
    result_file.close()
    
def run_ranking(basepath,topicIDs):
    for topicID in topicIDs:
        if os.path.exists(basepath+topicID+"/ideal.rank"):
            continue
        construct_rank_list(basepath, topicID)
    
if __name__=="__main__":
    print "start..."
    basepath = "/users/songwei/xuwenbin/diversification/ntcir10/rank1/"
    topics = os.listdir(basepath)
    topics.sort(cmp=None, key=None, reverse=False)
    
#     for topicID in topics:
#         print topicID
#         construct_rank_list(basepath,topicID)       
    threads = []
    topics1 = topics
    for i in range(5):
        ts1= []
        for j in range(len(topics1)):
            if int(topics1[j])%5== i:
                ts1.append(topics1[j])
        print ts1
      
        t1=  multiprocessing.Process(target=run_ranking,args=([basepath,ts1]))
        threads.append(t1)   
    for i in range(5):
        threads[i].start()
    for i in range(5):
        threads[i].join()
         

              
    
    print "End."