# -*- coding: utf8 -*-
import os
import pickle

from doc import getDocList
from invertedIndex import Index
from phraserepresentation import PhraseRepresentation
from querypreprocessing import getQueryListNoTopic
from wordrepresentation import WordRepresentation 


def getCandidatesRepresentations(queryfilename,topicfilename,docfilename,filepath,read_or_not):
    """getCandidatesRepresentations
    
    Parameters Return
    ----------
    candidateReps:查询候选
    cRepsForCluser:扩展的词和候选中的词的wordReps
    candembedding:通过字符串扩展的相似的词
    word2idWeight:词的权重和ＩＤ
    
    """
    topic_ID = queryfilename.split("/")[-1]
    pickle_path = filepath+"subtopic_cand_pickle/"
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    pickle_file = pickle_path+topic_ID
    if os.path.exists(pickle_file) and os.path.isfile(pickle_file):
        file_pick = open(pickle_file, 'rb')
        candidateReps=pickle.load(file_pick)
        cRepsForCluser=pickle.load(file_pick)
        candembedding= pickle.load(file_pick)
        file_pick.close()
        return candidateReps,cRepsForCluser ,candembedding
    else:
        
        doclist = getDocList(docfilename)
        iIndex = Index.BuildIndex(doclist)
        original_query, candidates, term_weight_sum, word2idWeight = getQueryListNoTopic(queryfilename, topicfilename, iIndex) # candidate in candidate = (label, wordlist, str)
        print "this is the original candidates,",len(candidates)
        candidateReps = []
        cRepsForCluser = {}
        emwordlist = []
        emworddict = {}
        word2Embeddingdict = {}
        wid = 0
        eminstance = WordRepresentation("", word2idWeight, iIndex)
        wordkeys = []
        candembedding={}
        
        for cand in candidates:
            words = cand[1]
            wordlist = []
            wordkeys.extend(words)
            candstr = cand[2]
            if candembedding.has_key(candstr):
                continue
            for word in words:
                if word.strip()=="":continue
                if word ==  original_query:continue
                if not emworddict.has_key(word):
                    emworddict[word] = wid
                    wid += 1
                wordlist.append(word)
                if word2Embeddingdict.has_key(word):
                    continue
                else:
                    wordrep =  WordRepresentation(word, word2idWeight, iIndex)
                    word2Embeddingdict[word] = wordrep.GetSimilarWordWeights()
            
            if len(wordlist)>0:
                wordweight = eminstance.GetSimilarWordWeights(wordlist)
                if wordweight == None:
                    continue
                for  term,m in wordweight:
                    emwordlist.append(term)
                    if not emworddict.has_key(term):
                        emworddict[term] = wid
                        wid +1
                candembedding[candstr]= wordweight
                
        for word in emworddict.keys():
            
            wordrep =  WordRepresentation(word, word2idWeight, iIndex)
            if wordrep.GetEmbeddingVec() != None and wordrep.GetCategoryVec()!=None:
                if not cRepsForCluser.has_key(word):
                    cRepsForCluser[word]=wordrep
                   
    
        candstrs_filter=set()
       
        for cand in candidates:
            label = cand[0]
            words = cand[1]
            candstr = cand[2]
            if candstr  in candstrs_filter:
                continue
            candstrs_filter.add(candstr)
            urlvec = cand[3]
            wordlist = []
            local_term_weight = []
            local_sum = 0.0
            wordreps = []
            docsum = 0.0
            for word in words:
                if word == original_query:
                    wordlist.append(word)
                    continue
                wordrep =  WordRepresentation(word, word2idWeight, iIndex)
                wordrep.SetEmbeddingwordlist(word2Embeddingdict[word])
                wordreps.append(wordrep)
                wordlist.append(word)
                local_term_weight.append(word2idWeight[word][1])
                local_sum += word2idWeight[word][1]
                doccount = len(iIndex.GetDocListByKey(word))*1.0
                docsum += doccount
            if docsum == 0.0:continue
            for i in range(len(local_term_weight)):
                local_term_weight[i] = (local_term_weight[i]/local_sum)#*(doccounts[i]/docsum)
              
            average_weight = [1.0 / len(local_term_weight) for i in range(len(local_term_weight))]
                     
                    
            phraseRep = PhraseRepresentation(label, candstr, wordreps,wordlist, local_term_weight, original_query,urlvec,emworddict,iIndex,cand[4]) 
            vsmVec = phraseRep.GetVSM() #phraseRep.GetEmbeddingVec()
            cateVec = phraseRep.GetCategoryVec()
            embeddingVec = phraseRep.GetEmbeddingVec()
            iDocVec = phraseRep.GetInvertedDocVec()
            URLVec = phraseRep.GetURLVec()
#             ewVec = phraseRep.GetEmbeddingWordVec()
            if vsmVec != None and cateVec != None and embeddingVec != None and iDocVec != None and URLVec != None:# and ewVec != None :
                candidateReps.append(phraseRep)
            else:
                print "candidate representation l126"
            
        print "prepared subtopic candidates:", len(candidateReps)
    
        output = open(filepath+"subtopic_cand_pickle/"+topic_ID, 'wb')
        pickle.dump(candidateReps, output)
        pickle.dump(cRepsForCluser,output)
        pickle.dump(candembedding,output)
        output.close()

    return candidateReps,cRepsForCluser ,candembedding#word2idWeight#similardict


if __name__ == '__main__':
    pass
