# -*- coding: utf-8 -*-

import string

from doc import getDocList
from invertedIndex import Index


def getStopWordList(fname):
    #获得停用词表
    f = open(fname)
    lines = f.readlines()
    f.close()
    stopwords = set()
    for line in lines:
        word = line.strip().replace('\n',"")
        stopwords.add(word)
     
    return stopwords

# this is a set rather a list
stopwords = getStopWordList("/users/songwei/xuwenbin/subtopic/ntcir09/stoplist_utf8.txt")

# Check whether a word is a good word.
def isGoodWord(word):
    try:
        term, pos = word.split('/')
    except:
        return False
    term = term.replace(' ', '')
    if term in stopwords or len(term) == 0 or (pos[0] not in ['n', 'v', 'j', 'i','w']) or (pos[0] == 'v' and len(term) == 1) or pos.strip() == "wp" or pos.strip()=="p" :
        return False
    return True

#本文件中的操作时对文件中的数据读取到内存中
#获得尽可能多的信息，包括词的集合及计数
def getQueryListNoTopic(queryfilename, topicfilename, iIndex): 
    
    selectedCandidates = []    
    term_weight_sum = 0
    word2idWeight = {}
    wid = 0
    topiclist = getTopicWordList(topicfilename)
    candidates,urlset = getQueryList(queryfilename)
    urllist = list(urlset)
    topictermset= set()
    #处理topiclist 将其合并成一个词，最后加上/q为标注
    ORIGINAL_QUERY=""
    for word in topiclist:
        term,pos = word.split("/")
        ORIGINAL_QUERY += term
        topictermset.add(term)
    ORIGINAL_QUERY += "/q"
    tstr = "".join(topiclist)
    
    filterSet = set()   
    candidates2= []
 
    for label, cand,count,urls in candidates:
        
        wordlist = []
        candstr = ""
        urlvec = [0.0 for i in range(len(urllist))]
        indexstart = 0
        flag = False
        #对文档进行处理要求的结果是候选必须包含原始查询中的所有词
        sum =0
        for term in topiclist:
            if cand.count(term)>0:
                sum+=1
                 
        if sum < len(topiclist)-2:
            continue
        for word in cand:
#             if word == topiclist[0]:
#                 indexstrat = cand.index(word)
#                 if indexstrat+len(topiclist)<len(cand)-1 :
#                      
#                     if  "".join(cand[indexstart:indexstart+len(topiclist)])== tstr:
#                        
#                         if indexstart!=0:
#                             temp = []
#                             temp.extend(cand[0:indexstart])
#                             temp.append(ORIGINAL_QUERY)
#                             candidates2.append([label,temp,count,urls])
#                         else:
#                             temp = []
#                             temp.append(ORIGINAL_QUERY)
#                             temp.extend(cand[len(topiclist):])
#                             candidates2.append([label,temp,count,urls])
            
            if word.split("/")[0] in topictermset:
                flag = True
        if flag == True:
            candidates2.append([label,cand,count,urls])
                
    print len(candidates2)
    for label, cand,count,urls in candidates2:               
        #放弃较长的候选            
        if len(cand) >8 or len(cand)<1:continue
        wordlist = []
        candstr = ""
        urlvec = [0.0 for i in range(len(urllist))]
        for word in cand:
            if word == ORIGINAL_QUERY:
                wordlist.append(word.split("/")[0])
                candstr += word.split("/")[0]
                continue
            
            if  not isGoodWord(word):
                candstr+= term
                continue
            
            term,pos = word.split("/")
            candstr +=term
            wordlist.append(term)
            if word2idWeight.has_key(term):
                word2idWeight[term][1]+=1
            else:
                word2idWeight[term]=[wid,1]
                wid +=1
            term_weight_sum +=1
        for url in urls:
            if url == 0:continue
            uid = urllist.index(url)
            urlvec[uid] += 1.0
        candstr = candstr.strip()
        if candstr.replace(" ", "") == ORIGINAL_QUERY.split("/")[0]:
            filterSet.add(candstr)
#         if candstr not in filterSet :
#             filterSet.add(candstr)
        selectedCandidates.append([label, wordlist, candstr,urlvec,count])
    return ORIGINAL_QUERY.split('/')[0], selectedCandidates, term_weight_sum, word2idWeight
     
    
# Extract key words from the original query
def getTopicWordList(filename):
    # 获得关键字列表
    f = open(filename)
    line = f.readline()
    f.close()
    words = line.strip().split('\t')
            
    return words

# Extract query reformulations into a list
def getQueryList(filename):
    #将查询文件中的查询导一个list中
    f = open(filename)
    lines = f.readlines()
    f.close()
   
    querylist = []   
    urlset = set()
    for line in lines:
#         print line
        elements = line.strip().split('||')
      
        if len(elements)!=4:continue
        label = int(elements[0])
        words = elements[1].split("\t")
        count = elements[2]
        if len(elements[3])>1:
            urls = elements[3].split("\t")
        else:
            urls = [elements[3].strip()]
        for url in urls:
            if url == 0:continue
            if url not in urlset:
                urlset.add(url)
        for word in words:
            if  not isGoodWord(word):
                words.remove(word)
       
        querylist.append((label, words,count,urls))
   
    return querylist,urlset
    
if __name__ == '__main__':
    doclist = getDocList("/users/songwei/xuwenbin/subtopic/ntcir11/doc/0001.txt")
    iIndex = Index.BuildIndex(doclist)
    original_query, candidates, term_weight_sum, word2idWeight=getQueryListNoTopic("/users/songwei/xuwenbin/subtopic/ntcir11/candidate/0001.txt","/users/songwei/xuwenbin/subtopic/ntcir11/topic/0001.txt",iIndex)
    print len(candidates)
   