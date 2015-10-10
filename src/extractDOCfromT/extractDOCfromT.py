# -*- coding:utf-8 -*-
import os

from build_evaluation_files_ntcir11 import eval_preprocessing


def readfilename(filename):
    print ""
    o = open(filename)
    lines = o.readlines()
    o.close()
    nameset = set()
    for line in lines:
        items = line.split(" ")
        if  items[2] not in nameset:
            
            nameset.add(items[2])
    return nameset

if __name__=="__main__":
    print "Start..."
    
#     results = "D:/diversification/ntcir10/ntcir10.Dqrels"
#     resultset = readfilename(results)
    outpath="D:/diversification/ntcir11/docs1/"
    docnames = eval_preprocessing("D:/diversification/ntcir11/")
    resultset = set(docnames)
    files = os.listdir("D:/diversification/page/")
    i = 0
    j= 0
    fileout = None
    BUFSIZE = 1024
    
    for file in files:
        print file
        o = open("D:/diversification/page/"+file,"rb")
        flag = False
        lines = o.readlines(BUFSIZE)
        while lines:
            
            for line in lines:
                if  line.count("<DOCNO>"):
                    print line
                i +=1 
                if line.startswith("<DOCNO>"):
                    
                    j+=1
                    if line[7:-9] in resultset:
                        flag = True
                        print line[7:-9]
                        resultset.remove(line[7:-9])
                        fileout = open(outpath+line[7:-9],"w")
                if line.startswith("</DOC>") :
                    flag = False
                    if fileout !=None and not fileout.closed :
                        fileout.close()
                
                if flag == True:
                    print >> fileout,line
            lines = o.readlines(BUFSIZE)
        o.close()
        
       
    