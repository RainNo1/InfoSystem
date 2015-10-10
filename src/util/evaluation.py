#-*- coding:utf8 -*-
import commands
import os


def call_eval_for_result(basepath,topicID,cutoff):
    """
        for each file get the best run name in the run list.
        parameter:inpath,topicID,cutoff
        return: the best docs name
    """
    
    #change the work directory and save the original work directory
    ori_directory = os.getcwd()
    
    if not ori_directory == basepath+"rank/"+topicID:
        os.chdir(basepath+"rank/"+topicID)
        
    #delete the useless files
    print topicID,"delete the useless files"
    files = os.listdir(basepath+"rank/"+topicID)
    file_filter = set()
    file_filter.add(topicID)
    file_filter.add("standard.Dqrels")
    file_filter.add("standard.Iprob")
    file_filter.add("standard.Iprob.tid")
    file_filter.add("ideal.rank")
    file_filter.add("runlist")
    with open(basepath+"rank/"+topicID+"/runlist") as lines:
        for line in lines:
            line=line.replace("\n","")
            file_filter.add(line)
        
    
    for name in files:
        if name not in file_filter and not name.lstrip().startswith(".nfs"):
            os.remove(basepath+"rank/"+topicID+"/"+name)
    if os.path.exists(basepath+"rank/"+topicID+"/"+topicID+"/"):
        files = os.listdir(basepath+"rank/"+topicID+"/"+topicID+"/")
        for name in files:
            if name.lstrip().startswith(".nfs"):continue
            if name.split(".")[-1]=="res" or name.split(".")[-1]=="glab":
                os.remove(basepath+"rank/"+topicID+"/"+topicID+"/"+name)

    shell_command = ""
    dqrels_file =  "standard.Dqrels"
    iprob_file =  "standard.Iprob"
    

    # built evaluation workspace
    #create Irelv , Grelv and din files
    print topicID,"start evaluation"
    shell_command = "/users/songwei/xuwenbin/eval/DIN-splitqrels "+iprob_file+" "+dqrels_file+" test"
    commands.getstatusoutput(shell_command)
    
    #create res files
    shell_command_res = "cat ./runlist | /users/songwei/xuwenbin/eval/TRECsplitruns ./standard.Iprob.tid"
    commands.getstatusoutput(shell_command_res)
    print topicID,"evaluation"
    #evaluate
    shell_command_eval = "cat ./runlist | /users/songwei/xuwenbin/eval/D-NTCIR-eval standard.Iprob.tid test "+str(cutoff)+" 110"
    commands.getstatusoutput(shell_command_eval)
    # evaluate the IA-ERR 
    shell_IA_eval = "cat ./runlist | /users/songwei/xuwenbin/eval/D-NTCIR-IAeval standard.Iprob.tid standard.Iprob test.Irelv 10 110"
    commands.getstatusoutput(shell_IA_eval)
    #reducing the original work directory
    os.chdir(ori_directory)
    return "Evaluation done."
if __name__=="__main__":
    print "start..."
    basepath = "/users/songwei/xuwenbin/diversification/ntcir09/"
    topicID = "0040"
    call_eval_for_result(basepath,topicID,10)