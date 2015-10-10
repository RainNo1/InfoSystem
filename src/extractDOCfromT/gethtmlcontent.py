# -*- coding: utf-8-*-
import os
import re


##过滤HTML中的标签
#将HTML中标签等信息去掉
#@param htmlstr HTML字符串.
def filter_tags(htmlstr):
    #先过滤CDATA
    re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
    re_script=re.compile(r'<script.[^>]*>.*?</script>', re.I|re.U|re.S)#Script
    re_style=re.compile(r'<style[^>]*>.*?</style>', re.I|re.U|re.S)#style
    re_br=re.compile('<br\s*?/?>')#处理换行
    re_h=re.compile('</?\w+[^>]*>')#HTML标签
    re_sp = re.compile("/\s+/g")
    re_comment=re.compile('<!--[^>]*-->')#HTML注释
    s=re_cdata.sub('',htmlstr)#去掉CDATA
    s=re_script.sub('',s) #去掉SCRIPT
    s=re_style.sub('',s)#去掉style
    s=re_br.sub('\n',s)#将br转换为换行
    s=re_h.sub('',s) #去掉HTML 标签
    s=re_comment.sub('',s)#去掉HTML注释
    #去掉多余的空行
    blank_line=re.compile(r'\n+')
    s=blank_line.sub('\n',s)
    s = re_sp.sub("",s)
    s=replaceCharEntity(s)#替换实体
    return s
##替换常用HTML字符实体.
#使用正常的字符替换HTML中特殊的字符实体.
#你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
#@param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES={'nbsp':' ','160':' ',
                'lt':'<','60':'<',
                'gt':'>','62':'>',
                'amp':'&','38':'&',
                'quot':'"','34':'"',}
    re_charEntity=re.compile(r'&#?(?P<name>\w+);')
    sz=re_charEntity.search(htmlstr)
    while sz:
        entity=sz.group()#entity全称，如&gt;
        key=sz.group('name')#去除&;后entity,如&gt;为gt
        try: #by www.jbxue.com
            htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
            sz=re_charEntity.search(htmlstr)
        except KeyError:
            #以空串代替
            htmlstr=re_charEntity.sub('',htmlstr,1)
            sz=re_charEntity.search(htmlstr)
   
    str=""
    for it in htmlstr.split("\n"):
        it.replace(" ","")
        if it.strip()!="":
            str +=it.strip()+"\n"
            
    return str
def repalce(s,re_exp,repl_string):
    return re_exp.sub(repl_string,s)
if __name__=='__main__':
    path = "D:/diversification/ntcir11/documents/"
    pathouth = "D:/diversification/ntcir11/documents_pre/"
    filepathin = os.listdir(path)
    for pathin in filepathin:
        print pathin
        pathout = pathouth + pathin+"/"
        if not os.path.exists(pathout):
            os.makedirs(pathout)
        files = os.listdir(path+pathin+"/")
        for filename in files:
            
            fileout = open(pathout+filename,"w")
            s = file(path +pathin+"/"+filename).read()
            str = filter_tags(s)
            print >> fileout,str
            fileout.close()
            
            
   