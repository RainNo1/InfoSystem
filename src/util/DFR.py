#-*- coding:utf8 -*-
import math


class DFRsimilar:
    def __init__(self):
        self.field = ""
        self.numberofDocuments=0
        self.numberofFieldTokens=0
        self.avgFieldLength = 0
        self.docFreq = 0
        self.totalTermFreq=0
        self.queryBoost = 0.0
        self.topLevelBoost = 0.0
        
    def GetScore(self,query,doclist):
        print""
        tfn = self.normaliztionH1(self,self.avgFieldLength,self.totalTermFreq,self.numberofDocuments)
        sim = self.queryBoost * self.score_if(self.totalTermFreq, tfn*self.score_afterEffectL(tfn))
        return sim
    def score_if(self,num_docs,totalfre_term,tfn):
        print ""
        return tfn*math.log(1+(num_docs+1)/(totalfre_term+0.5),2)
    def score_afterEffectL(self,tfn):
        return 1 / (tfn + 1)
    def normaliztionH1(self,aveFiledLength,tf,lens):
        return tf * aveFiledLength / lens;
if __name__=="__main__":
    print "start..."