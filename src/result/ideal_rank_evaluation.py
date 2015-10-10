#-*- coding:utf8 -*-
import os
import sys
sys.path.append("..")
from result.result_analysis import DR_result_analysis
from util.evaluation import call_eval_for_result





if __name__=="__main__":
    print "start..."
    files = os.listdir("/users/songwei/xuwenbin/diversification/ntcir09/rank/")
    for file_name in files:
        out1 = open("/users/songwei/xuwenbin/diversification/ntcir09/rank/"+file_name+"/runlist","w")
        out = open("/users/songwei/xuwenbin/diversification/ntcir09/rank/"+file_name+"/ideal","w")
        print >>out1,"ideal"
        with open("/users/songwei/xuwenbin/diversification/ntcir09/rank/"+file_name+"/ideal.rank") as lines:
            for line in lines:
                if line.startswith("0"):
                    print>>out,line.replace("\n","")
        out.close()
        out1.close()
        call_eval_for_result("/users/songwei/xuwenbin/diversification/ntcir09/",file_name,10) 
    DR_result_analysis("/users/songwei/xuwenbin/diversification/ntcir09/","","ideal")