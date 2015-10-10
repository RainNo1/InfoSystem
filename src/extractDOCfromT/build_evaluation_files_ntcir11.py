#-*- coding:utf8 -*-
import os
from xml.etree.ElementTree import Element

import xml.etree.cElementTree as ET


# e = Element()
# e.text
def parser_doc_Xml(filename):
    tree = ET.ElementTree(file=filename)
    root =  tree.getroot()
    result = {}
    for relevance in root:
        for doc in relevance:
            doc_id = doc.attrib["docid"]
            query_id = doc.attrib["queryid"]
            doc_query_relevance = doc.attrib["relevance"]
            subtopic =doc.attrib["sls"]
            if result.has_key(query_id):
                if result[query_id].has_key(doc_id):
                    result[query_id][doc_id].append([doc_query_relevance,subtopic])
                else:
                    result[query_id][doc_id] = []
                    result[query_id][doc_id].append([doc_query_relevance,subtopic])
            
            else:
                result[query_id]={}
                result[query_id][doc_id] = []
                result[query_id][doc_id].append([doc_query_relevance,subtopic])
    
    return result
           
def parser_subtopic_Xml(filename):
    """
    parameter: filename
    return: a dictionary for build the Iprob files and connect query to query-id
    """
    tree = ET.ElementTree(file=filename)
    root =  tree.getroot()
    result ={}#{topicID:[{label:poss},{src:label}]}
    
    for topic in root:
        topicID = topic.attrib["id"]
        if not result.has_key(topicID):
            result[topicID]=[{},{}]#[先知_电影_概况:[1,0.09],先知影评：1]
            
        for fls in topic:
            for items in fls:# items in first level subtopic [ex,second,second...] 
                if items.tag == "examples":continue
                subtopic_label = items.attrib["content"]
                subtopic_poss = items.attrib["poss"]
                if result.has_key(topicID):
                    if not result[topicID][0].has_key(subtopic_label):
                        result[topicID][0][subtopic_label] = subtopic_poss
                        
                    
                for item in items:# item in second level subtopics [ex,ex,ex...]
                    subtopic_cand =  item.text
                    if  not result[topicID][1].has_key(subtopic_cand):
                        result[topicID][1][subtopic_cand] = subtopic_label
       
    return result
                    
def eval_preprocessing(basepath):
    subtopic_xml = basepath+"IMine.Qrel.SMC.xml"
    doc_xml = basepath+ "IMine.Qrel.DRC.xml"
    iprob_file = open(basepath+"ntcir11.Iprob","w")
    dqrel_file = open(basepath+"ntcir11.Dqrels","w")
    subtopic_dict = parser_subtopic_Xml(subtopic_xml)
    document_dict = parser_doc_Xml(doc_xml)
    topicIDs = subtopic_dict.keys()
    topicIDs.sort()
    documents =[]
    # solving the label to id problem
    for topicID in topicIDs[:-1]:
        print topicID
        subtopic2id = {}
        items = subtopic_dict[topicID][0].items()
        items.sort(cmp=None,key=lambda asd:asd[1],reverse=True)
        for subtopic_index,item in enumerate(items):
            if subtopic2id.has_key(item[0]):
                continue
            else:
                subtopic2id[item[0]]=[subtopic_index+1,item[1]]
        construct_Iprob(items,iprob_file,topicID)
         
        documents.extend(document_dict[topicID].keys())
        construct_Dqrels(document_dict[topicID],subtopic2id,dqrel_file,topicID,basepath)
    return documents
    


def construct_Dqrels(documents,subtopic2id,dqrel_file,topicID,basepath):
    document_original = os.listdir(basepath+"documents_seg/"+topicID)
    for document_id in document_original:
        if documents.has_key(document_id):
            for dqrel,subtopic in documents[document_id]:
                rel = "L0"
                if dqrel == "2":
                    rel = "L2"
                elif dqrel == "3":
                    rel = "L3"
                 
                subtopicID = subtopic2id[subtopic][0]
                print>>dqrel_file,"%s %d %s %s"%(topicID,subtopicID,document_id,rel)
        else:
            print>>dqrel_file,"%s 0 %s L0"%(topicID,document_id)  
   
        
            
        
def construct_Iprob(subtopic_items,iprob_file,topicID):
    ""
    for subtopic_index,item in enumerate(subtopic_items):
        print >>iprob_file,"%s %d %s"%(topicID,subtopic_index+1,item[1])
    


if __name__=="__main__":
    print "start..."
    basepath = "D:/diversification/ntcir11/"
    eval_preprocessing(basepath)