# -*- coding: utf-8 -*-
from document import Document


def getDocList(filename):
    f = open(filename)
    doclist = [] 
    line = f.readline()
    i =0   
    while line:
       
        if i==0:
            doc = Document()
            doc.SetId(line[:-2])
        else:
            if i == 1:
                words = line.strip().split('\t') 
                doc.SetLabel(words)
            else:
                words = line.strip().split('\t')
                doc.SetDocStr(line)
                terms = []
                pos = []
                for word in words:
                    term_pos = word.split("/")
                    if len(term_pos) == 2:
                        terms.append(term_pos[0])
                        pos.append(term_pos[1])
                for word in doc.GetLabel():
                    term_pos = word.split("/")
                    if len(term_pos) == 2:
                        terms.append(term_pos[0])
                        pos.append(term_pos[1])
                doc.SetTermVec(terms)
                doc.SetPOSVec(pos)
                doclist.append(doc)
               
                                      
        i +=1
        i = i%3
        
        line = f.readline()
    
    f.close()
    
    return doclist

if __name__ == '__main__':
    docs = getDocList("D:/doc/4.txt")
    print "".join(docs[1].GetTermVec())
   
