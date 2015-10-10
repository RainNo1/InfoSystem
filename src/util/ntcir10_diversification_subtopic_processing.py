#-*- coding:utf8 -*-
import os
import shutil


if __name__=="__main__":
    print "start..."
    inpath = "D:/diversification/ntcir10/"
    outpath = "D:/diversification/ntcir10/"
    # processing the candidate to subtopic
    
#     for filename in os.listdir(inpath):
#         infile = open(inpath+filename)
#         outfile = open(outpath+filename.split(".")[0],"w")
#         for line in infile.readlines():
#             line =  line.replace("\n","")
#             items = line.split("\t")
#             label = items[0]
#             subtopic = items[1:-1]
#             count = "1"
#             url ="0"
#             print>>outfile,label+"||"+"\t".join(subtopic)+"||"+count+"||"+url

    #processing the topic into standard format
    for filename in os.listdir(inpath+"topic1"):
        
        shutil.copy(inpath+"topic1/"+filename,outpath+"topic/"+filename.split(".")[0])