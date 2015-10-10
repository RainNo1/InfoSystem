# -*- coding:utf-8 -*-
import os
import shutil

def readdocdict(filename):
    o = open(filename)
    lines = o.readlines()
    o.close()
    docdict = {}
    i = 0
    for line in lines:
        items = line.split(" ")
        if docdict.has_key(items[0]):
            docdict[items[0]].append(items[2])
            i+=1
        else:
            docdict[items[0]] = []
            docdict[items[0]].append(items[2])
            i+=1
    print i
    return docdict

    
if __name__=="__main__":
    docdict = readdocdict("D:/diversification/ntcir10/ntcir10.Dqrels")
    pathin = "D:/diversification/ntcir10/docs/"
    i = 0
    for key in docdict.keys():
        print key
        path = "D:/diversification/ntcir10/documents/"+key+"/"
        print path
        if not os.path.exists(path):
            print path
            os.makedirs(path)
        for file in docdict[key]:
            
            if os.path.exists(pathin+file):
                i+=1
                shutil.copyfile(pathin+file,path+file)
    print i