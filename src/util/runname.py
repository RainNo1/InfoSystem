#-*- coding:utf8 -*-

def build_run_name(dictionary_loss,dictionary_norm,dictionary_rep,document_ori_rep,document_rep_norm,mine_method):
    runname = "SR"
    if mine_method =="1":
        runname = "SR"
    else:
        runname = "MR"
    if dictionary_loss == 'lasso_lars':
        runname +="LL"
    elif dictionary_loss =="lasso_cd":
        runname += "LC"
    elif dictionary_loss =="lars":
        runname += "LS"
    elif dictionary_loss == "omp":
        runname += "OM"
    elif dictionary_loss == "threshold":
        runname += "TH"
    else:
        return None
    
    
    runname += dictionary_rep
    runname += dictionary_norm
    runname +=document_ori_rep
    runname += document_rep_norm
    return runname

def construct_doc_str(runname):
    doc_str = "this run is based sparse representation:\n the parameters:\n loss function:"
    if runname[2:4] == "LL":
        doc_str += "lasso_lars\n"
    elif runname[2:4] == "LC":
        doc_str += "lasso_cd\n"
    elif runname[2:4] == "LS":
        doc_str += "lars\n"
    elif runname[2:4] == "OM":
        doc_str += "omp\n"
    elif runname[2:4] == "TH":
        doc_str += "threshold\n"
    doc_str += "subtopic (dictionary) representation:"
    if runname[4:6] == "S1":
        doc_str += "tf score addition,no term weight consideration\n"
    elif runname[4:6] == "S2":
        doc_str += "tf score addition,term weight:subtopic frequency of the term in this subtopic set /the subtopic frequency of the term in collection \n"
    elif runname[4:6] == "E1":
        doc_str += "dictionary construct by the expansion words,and expansion processing no consideration with the original query top 8 words\n"
    elif runname[4:6] == "E2":
        doc_str += "dictionary construct by the expansion words,and expansion processing consider the original query,top 8 words \n"
    elif runname[4:6] == "E3":
        doc_str += "dictionary construct by the expansion word clusters which match the query,and expansion processing no consideration with the original query,top 15 words\n"
    elif runname[4:6] == "E4":
        doc_str += "dictionary construct by the expansion word clusters which match the query,and expansion processing consider the original query,top 15 words \n"
    elif runname[4:6] == "E5":
        doc_str += "dictionary construct by the expansion word of the query+original words,and expansion processing no consideration with the original query,top 5 words \n"
    elif runname[4:6] == "E6":
        doc_str += "dictionary construct by the expansion word of the query+original words,and expansion processing consider the original query,top 5 words \n"
    doc_str += "subtopic original representation is Normalization:"
    if runname[6:7] == "Y":
        doc_str += "yes\n"
    elif runname[6:7] == "N":
        doc_str += "no\n"
    doc_str +="document original representation:"
    if runname[7:9] == "D1":
        doc_str += " the tf/idf score is used to build the original representation.no other weight in consideration \n"
    elif runname[7:9] == "D2":
        doc_str += "the tf/idf and the number of subtopic contain this term/the number of subtopic set are used as term weight \n"
    elif runname[7:9] == "D3":
        doc_str += "the tf/idf and(the idf score of the term in subtopic collection) are used as term weight \n"
    elif runname[7:9] == "D4":
        doc_str += "the tf/idf and(original query word count in collection/ the term count in collection) are used as term weight \n"
    elif runname[7:9] == "D5":
        doc_str += "the tf/idf and(original query word count in collection/ the document frequency of the term ) are used as term weight \n"
    doc_str += "document original representation is Normalization:"
    if runname[9:10] == "Y":
        doc_str += "yes\n"
    elif runname[9:10] == "N":
        doc_str += "no\n"
    doc_str+= "sparse representation balance parameter_1:"+runname[10:12]+"\n"
    doc_str+= "subtopic collection is complete:"
    if runname[12:14]=="LY":
        doc_str+="not complete\n"
    else:
        doc_str += "is complete\n"
    if runname[14:16] == "N1":
        doc_str+="without dictionary learning \n sparse coding with cnu non-negative\n"
    elif runname[14:16] == "N2":
        doc_str+="without dictionary learning \n sparse coding with sklearn non-negative\n"
    elif runname[14:16] == "N3":
        doc_str+="without dictionary learning \n sparse coding with cnu original\n"
    elif runname[14:16] == "N4":
        doc_str+="without dictionary learning \n sparse coding with sklearn original\n"
    
    elif runname[14:16]=="L1":
        doc_str+="dictionary learning with  learning1 init-coding  \n sparse coding with cnu non-negative\n"
    elif runname[14:16]=="L2":
        doc_str+="dictionary learning with  learning2 %2==0:init-coding \n sparse coding with cnu non-negative\n"
    elif runname[14:16]=="L3":
        doc_str+="dictionary learning with  learning1 init-coding  \n sparse coding with cnu original\n"
    elif runname[14:16]=="L4":
        doc_str+="dictionary learning with  learning2 %2==0:init-coding \n sparse coding with cnu original\n"
    
    
    elif runname[14:16]=="L5":
        doc_str+="dictionary learning with  learning1 init-coding  \n sparse coding with sklearn + non-negative\n"
    elif runname[14:16]=="L6":
        doc_str+="dictionary learning with  learning2 %2==0:init-coding \n sparse coding with sklearn+ non-negative\n"
    elif runname[14:16]=="L7":
        doc_str+="dictionary learning with  learning1init-coding  \n sparse coding with cnusklearn +original\n"
    elif runname[14:16]=="L8":
        doc_str+="dictionary learning with  learning2%2==0:init-coding  \n sparse coding with sklearn +original\n"
    doc_str+= "sparse representation balance parameter_2:"+runname[16:18]+"\n"
    doc_str+= "sparse representation balance parameter_3:"+runname[18:]
    return doc_str

if __name__=="__main__":
    print "start..."