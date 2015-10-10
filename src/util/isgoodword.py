#-*- coding:utf8 -*-

def getStopWordList(fname):
    #getStop word from file stopwords_cn.dat
    f = open(fname)
    lines = f.readlines()
    f.close()
    stopwords = set()
    for line in lines:
        word = line.strip().replace('\n',"")
        if word not  in stopwords:
            stopwords.add(word)
     
    return stopwords

# this is a set rather a list
# stopwords = getStopWordList("D:/diversification/Stopwords_cn.dat")
stopwords = getStopWordList("/users/songwei/xuwenbin/diversification/Stopwords_cn.dat")

# Check whether a word is a good word.
def isGoodWord(word):
    try:
        term, pos = word.split('/')
    except:
        return False
    term = term.replace(' ', '')
    term = term.strip()
    
    if term in stopwords or len(term) == 0 or (pos[0] not in ['n', 'v', 'j', 'i','w']) or (pos[0] == 'v' and len(term) == 1) or pos.strip() == "wp" or pos.strip()=="p" :
        return False
   
    return True
if __name__=="__main__":
    print "start..."
    
  