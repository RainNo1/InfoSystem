#-*- coding:utf8 -*-
import os

from base_model.document import Document
from util.isgoodword import isGoodWord


# sys.path.append("..")
def doc_preprocessing(basepath,topicID,word2id_weight,topic_words):
    """
    """
    document_path = basepath+"documents_seg/" + topicID +"/"
    documents = os.listdir(document_path)
    doc_subtopic_path = basepath+"doc_subtopic_relation/"
    
    #store information of term in the subtopic word set.
    #include  docs ID which term appear and term frequency in collection.
    document_term_frequency = {}
    document_list = []

    sum_document_len=0.0
    topic_word_count=0.0
#     true_rank = get_document_true_rank(basepath,topicID)
    doc_sid = get_document_relate_subtopicID(doc_subtopic_path , topicID)
    for document in documents:
#         if not true_rank.has_key(document):
#             continue
        o = open(document_path+document)
        lines = o.readlines()
        o.close()
        
        terms =[]
        terms_pos = []
        for line in lines:
            line = line.replace("\n","")
            words = line.split("\t")
            
            for windex,word in enumerate(words):
                if word in topic_words :
                    topic_word_count +=1
                if isGoodWord(word):
                    sum_document_len+=1
                else:
                    continue
                term,pos = word.split("/")
                if document_term_frequency.has_key(term):
                    document_term_frequency[term][0] += 1
                    if document_term_frequency[term][1].count(document)==0:
                        document_term_frequency[term][1].append(document)
                else:
                    item = [1,[document]]
                    document_term_frequency[term]=item
                if word2id_weight.has_key(term):
                    terms.append(term)
                    terms_pos.append(pos)
                    
#         sum_document_len+= len(terms)
        doc = Document()
        doc.set_doc_str(" ".join(terms))
        doc.set_id(document)
        doc.set_term_vec(terms)
        doc.set_pos_vec(terms_pos)
        doc.set_true_rank("1")
        doc.set_related_sid(doc_sid[document])
        document_list.append(doc)
        
    average_document_len = sum_document_len/len(document_list)
                    
    return document_list,document_term_frequency,average_document_len,topic_word_count
                

def get_document_true_rank(basepath,topicID):
    """
    """
    true_rank_file = basepath+"rank/"+topicID+"/ideal.rank"
    
    true_rank={}

    o = open(true_rank_file)
    lines = o.readlines()
    o.close()
    
    for line in lines:
        line = line.replace("\n","")
        items = line.split(" ")
        if len(items)==6:
            if not true_rank.has_key(items[2]):
                true_rank[items[2]] = items[3]
    return true_rank



def get_document_relate_subtopicID(filepath,topicID):
    doc_sid = {}    # a dictionary store the docs and related IDs
    o = open(filepath+topicID)
    lines = o.readlines()
    o.close()
    for line in lines:
        line = line.replace("\n","")
        items = line.split(" ")
        doc_id = items[2]
        if doc_sid.has_key(doc_id):
            doc_sid[doc_id].append([items[1],items[3]])
        else:
            doc_sid[doc_id] =[[items[1],items[3]]]
    return doc_sid                
                    
            
            
        
if __name__=="__main__":
    print "start..."
    print get_document_relate_subtopicID("D:/diversification/ntcir09/doc_subtopic_relation/","0001")