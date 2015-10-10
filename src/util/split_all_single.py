#-*- coding:utf8 -*-
import os
import sys
#split the Dqrels and Iprob files to single one and named as the topicID
def spilt_file(path,filein,pathout):
    fin = path + filein

    name_suffix=filein.split(".")[1]
    with open(fin) as lines:
        filename =""
        out = None
        for line in lines:
            items = line.split(" ")
            if items[0]== filename:
                out.write(line)
            else:
                filename = items[0]
                if out!= None and not out.closed:
                    out.close()
                if not os.path.exists(pathout+filename):
                    os.mkdir(pathout+filename)
                out = open(pathout+filename+"/standard."+name_suffix,"w")
                out.write(line)
def build_document_subtopic_relation_files(path,filein,pathout):
    fin = path + filein

    with open(fin) as lines:
        filename =""
        out = None
        for line in lines:
            print line
            items = line.split(" ")
            if items[0]== filename:
                out.write(line)
            else:
                filename = items[0]
                if out!= None and not out.closed:
                    out.close()
               
                out = open(pathout+filename,"w")
                out.write(line)
            
if __name__=="__main__":
    print "start..."
    path = "/users/songwei/xuwenbin/diversification/ntcir10/"
    pathout="/users/songwei/xuwenbin/diversification/ntcir10/rank/"

    filename = "ntcir10.Iprob"
    spilt_file(path,filename,pathout)

    filename = "ntcir10.Dqrels"
    spilt_file(path,filename,pathout)

#     filename = "ntcir10.Dqrels"
#     pathout="D:/diversification/ntcir10/doc_subtopic_relation/"
#     build_document_subtopic_relation_files(path, filename,pathout)

