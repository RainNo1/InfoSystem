#-*- coding:utf8 -*-
import os
def result_anaylysis_second_step(basepath):
    """
    for all method run systems,rank the run result by three score.
    """
    files = os.listdir(basepath)
    ndcg=[]
    irec = []
    dq = []
    dcg=[]
    ia = []
    for filename in files:
        with open(basepath+filename) as lines:
            for line in lines:
                if line.startswith("I-rec@0010="):
                    items = line.split("=")
                    score = float(items[1].lstrip())
                    irec.append([filename,score])
                if line.startswith("nDCG@0010="):
                    items = line.split("=")
                    score = float(items[1].lstrip())
                    dcg.append([filename,score])
                if line.startswith("D#-Q@0010="):
                    items = line.split("=")
                    score = float(items[1].lstrip())
                    dq.append([filename,score])
                if line.startswith("D#-nDCG@0010= "):
                    items = line.split("=")
                    score = float(items[1].lstrip())
                    ndcg.append([filename,score])
                if line.startswith("IA-nERR@0010="):
                    items = line.split("=")
                    score = float(items[1].lstrip())
                    ia.append([filename,score])
    ndcg.sort(key=lambda x:x[1],reverse=True)
    dq.sort(key=lambda x:x[1],reverse=True)
    irec.sort(key=lambda x:x[1],reverse=True)
    dcg.sort(key=lambda x:x[1],reverse=True)
    ia.sort(key=lambda x:x[1],reverse=True)
    output = open(basepath+"result.anaylysis","w")
    print >>output,"-----------the-best@10------------\n"
    print >>output,("%0.4f  %0.4f %0.4f %0.4f")%(irec[0][1],dcg[0][1],ndcg[0][1],ia[0][1])
    print >>output,"\n\n-----------D#-nDCG@0010------------"
    for filename,socre in ndcg:
        print>>output, filename,socre
        
#     print >>output,"-----------D#-Q@0010---------------"
#     for filename,socre in dq:
#         print>>output, filename,socre
        
        
    print >>output,"-----------I-rec@0010--------------"
    for filename,socre in irec:
        print>>output, filename,socre
    print >>output,"-----------nDCG@0010--------------"
    for filename,socre in dcg:
        print>>output, filename,socre
    print >>output,"-----------IA-ERR@0010--------------"
    for filename,socre in ia:
        print>>output, filename,socre
def combine_result(basepath):   
    output = open(basepath+"result_SDR","w")
    items = []
    for child_path in os.listdir(basepath):
        if os.path.isfile(basepath+child_path):continue
        lines = open(basepath+child_path+"/result.anaylysis").readlines()
        for line in lines:
            if line.startswith("0"):
                print line
                filename = "_".join(child_path.split("_")[:-1])
                method = child_path.split("_")[-1]
                line = line.replace("\n","")
              
                items.append([filename,method,line])
                
                
    items.sort(cmp=None, key=lambda asd:asd[1])
    for item in items:
        if item[0]=="SRD_mine_all" :
            print >>output,item[0]+"_"+item[1]+"\t\t"+item[-1]
        else:
            print >>output,item[0]+"_"+item[1]+"\t"+item[-1]
        
    
    
    
    
    
     
if __name__=="__main__":
    print "start..."
#     basepath = "C:/Users/dell/Desktop/result_09/"
#     for child_path in os.listdir(basepath):
#         
#         if os.path.isfile(basepath+child_path):continue
#         result_anaylysis_second_step(basepath+child_path+"/")
#     combine_result(basepath)    
    
    basepath = "C:/Users/dell/Desktop/result_10/"
    for child_path in os.listdir(basepath):
        if os.path.isfile(basepath+child_path):continue
        result_anaylysis_second_step(basepath+child_path+"/")
    combine_result(basepath)