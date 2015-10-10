#-*- coding:utf8 -*-
import os


def result_analysis_sparse(file_in,fileout):
    items = []
    out = open(fileout,"w")
    s1 =s2= s3= 0.0
    i = 0
    with open(file_in) as lines:
        for line in lines:
            i+=1
            line = line.replace("\n","")
            item=line.split("\t")
            s1 += float(item[2])
            s2 +=float(item[3])
            s3 += float(item[4])
#             if float(item[1])<float(item[3])+float(item[4]):
#                 continue
            items.append(item)
            
    items.sort(cmp=None, key=lambda asd:int(asd[0]), reverse=False)
    for item in items:
        
        print>>out,"\t".join(item)
    print>>out,"\t\t%2.4f\t%2.4f\t%2.4f"%(s1/i,s2/i,s3/i)
    out.close()
    
if __name__=="__main__":
    print "start..."
    
    basepath = "D:/diversification/ntcir09/results/2014120401/middle/"
    topic_ids = os.listdir(basepath)
    topic_ids.sort()
    for topic_id in topic_ids:
        print topic_id
        if topic_id =="0000":continue
        result_analysis_sparse(basepath + topic_id+"/SRLCE5ND5N06.out",basepath+"0000/"+topic_id+"SRLCE5ND5N06.sort")
       
    
    
    
    