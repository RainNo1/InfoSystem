#-*- coding:utf8 -*-
import os


def DR_result_analysis(basepath,doc_str,result_name):
    """
     I-rec@0050=          0.9000
     D#-Q@0050=           0.5942
     D#-nDCG@0050=        0.7130
     nDCG@0010=

    """
    filepath = basepath+"rank/"
    result = open(result_name+".result","w")
    files = os.listdir(filepath)
    print >> result,doc_str
    Irec_score = 0.0
    Irec_num = 0
    DQ_score = 0.0
    DQ_num = 0
    nDCG_score = 0.0
    nDCG_num = 0
    DCG_score = 0.0
    DCG_num = 0
    IA_num = 0
    IA_score = 0.0
    files.sort()
    print result_name,"analysis processing"
    for topicID in files:
        
        o = open(filepath+topicID+"/"+result_name+".110.Dnev")
        lines = o.readlines()
        o.close()
        
        for line in lines:
            items = line.split(" ")
            if items[1] =="I-rec@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
#                     Irec_num -=1
                    continue
                Irec_score += float(items[-1].replace("\n",""))
                Irec_num += 1
            if items[1] =="D#-Q@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                DQ_score += float(items[-1].replace("\n",""))
                DQ_num +=1
            if items[1] =="D#-nDCG@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                nDCG_score += float(items[-1].replace("\n",""))
                nDCG_num +=1
            if items[1] =="nDCG@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                DCG_score += float(items[-1].replace("\n",""))
                DCG_num +=1
       
        o = open(filepath+topicID+"/"+result_name+".110.nevIA")
        lines = o.readlines()
        o.close()
        for line in lines:
            items = line.split(" ")
            if items[3] =="IA-nERR@0010=" :
                print >>result,items[0]+" "+" ".join(items[3:])
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                IA_score += float(items[-1].replace("\n",""))
                IA_num += 1
#     print Irec_num,DQ_num,nDCG_num,DCG_num,IA_num
    print >>result,"average:"
    print >>result,"-----------------------------------------"
    print >>result,"I-rec@0010=        %s" % str(Irec_score/Irec_num)
    print >>result,"D#-Q@0010=         %s" % str(DQ_score/DQ_num)
    print >>result,"D#-nDCG@0010=      %s" % str(nDCG_score/nDCG_num)
    print >>result,"nDCG@0010=      %s" % str(DCG_score/DCG_num)
    print >>result,"IA-nERR@0010=      %s" % str(IA_score/IA_num)
    result.close()
def DR_result_analysis_sparse(basepath,doc_str,result_name,collection):
    """
     I-rec@0050=          0.9000
     D#-Q@0050=           0.5942
     D#-nDCG@0050=        0.7130
     nDCG@0010=

    """
    filepath = basepath+"rank/"
    if not os.path.exists("/users/songwei/xuwenbin/diversification/src/ranking/"+collection+"/"):
        os.mkdir("/users/songwei/xuwenbin/diversification/src/ranking/"+collection+"/")
    result = open("/users/songwei/xuwenbin/diversification/src/ranking/"+collection+"/"+result_name+".result","w")
    files = os.listdir(filepath)
    print >> result,doc_str
    Irec_score = 0.0
    Irec_num = 0
    DQ_score = 0.0
    DQ_num = 0
    nDCG_score = 0.0
    nDCG_num = 0
    DCG_score = 0.0
    DCG_num = 0
    IA_num = 0
    IA_score = 0.0
    files.sort()
    print result_name,"analysis processing"
    for topicID in files:
        if not os.path.exists(filepath+topicID+"/"+result_name+".110.Dnev"):continue
        o = open(filepath+topicID+"/"+result_name+".110.Dnev")
        lines = o.readlines()
        o.close()
        
        for line in lines:
            items = line.split(" ")
            if items[1] =="I-rec@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
#                     Irec_num -=1
                    continue
                Irec_score += float(items[-1].replace("\n",""))
                Irec_num += 1
            if items[1] =="D#-Q@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                DQ_score += float(items[-1].replace("\n",""))
                DQ_num +=1
            if items[1] =="D#-nDCG@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                nDCG_score += float(items[-1].replace("\n",""))
                nDCG_num +=1
            if items[1] =="nDCG@0010=" :
                print >>result,line
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                DCG_score += float(items[-1].replace("\n",""))
                DCG_num +=1
       
        o = open(filepath+topicID+"/"+result_name+".110.nevIA")
        lines = o.readlines()
        o.close()
        for line in lines:
            items = line.split(" ")
            if items[3] =="IA-nERR@0010=" :
                print >>result,items[0]+" "+" ".join(items[3:])
                if  float(items[-1].replace("\n",""))>1.0 or float(items[-1].replace("\n",""))==0:
                    continue
                IA_score += float(items[-1].replace("\n",""))
                IA_num += 1
#     print Irec_num,DQ_num,nDCG_num,DCG_num,IA_num
    print >>result,"average:"
    print >>result,"-----------------------------------------"
    print >>result,"I-rec@0010=        %s" % str(Irec_score/Irec_num)
    print >>result,"D#-Q@0010=         %s" % str(DQ_score/DQ_num)
    print >>result,"D#-nDCG@0010=      %s" % str(nDCG_score/nDCG_num)
    print >>result,"nDCG@0010=      %s" % str(DCG_score/DCG_num)
    print >>result,"IA-nERR@0010=      %s" % str(IA_score/IA_num)
    result.close()

        
if __name__=="__main__":
    print "start..."
    DR_result_analysis("/users/songwei/xuwenbin/diversification/ntcir09/")
    