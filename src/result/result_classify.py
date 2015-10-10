#-*- coding:utf8 -*-
import os
import shutil

def classify_result(basepath,basepath_out,mine,less,second_path):
    for filename  in os.listdir(basepath):
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="N1":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="N2":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="N3":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="N4":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L1":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L2":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L3":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L4":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L5":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L6":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L7":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
        if filename[0]==mine and filename[12:14]==less and filename[14:16]=="L8":
            if not os.path.exists(basepath_out+second_path+filename[14:16]+"/"):
                os.mkdir(basepath_out+second_path+filename[14:16]+"/")
            shutil.move(basepath+filename,basepath_out+second_path+filename[14:16]+"/"+filename)
            
        
            
            
            
            
            
     
    
if __name__=="__main__":
    print "start..."

    items = []
    items.append(["S","LN","SRD_standard_all_"])
    items.append(["S","LY","SRD_standard_less_"])
    items.append(["M","LN","SRD_mine_all_"])
    items.append(["M","LY","SRD_mine_less_"])
    
    basepath = "C:/Users/dell/Desktop/ntcir10/"
    basepath_out = "C:/Users/dell/Desktop/result_10/"
    for mine,less,second_path in items:
        classify_result(basepath, basepath_out, mine, less, second_path)
    
#     basepath = "C:/Users/dell/Desktop/ntcir09/"
#     basepath_out = "C:/Users/dell/Desktop/result_09/"
#     for mine,less,second_path in items:
#         classify_result(basepath, basepath_out, mine, less, second_path)
    
        









