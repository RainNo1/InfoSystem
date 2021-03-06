import math

from numpy import array, zeros, append
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity



# input a list of feature vector
# output a squre matrix
def precomputing_distance(arraylist):
    size = len(arraylist)
    distance_matrix = zeros((size, size), float)
    sim_matrix = cosine_similarity(arraylist)
    #print sim_matrix
    sumd = 0.0
    for i in range(size):
        for j in range(i, size):
            #print sim_matrix[i][j]
            distance_matrix[i][j] = distance_matrix[j][i] = 1 - sim_matrix[i][j]
            sumd += sim_matrix[i][j]*2
#     for i,avec in enumerate(arraylist):
#         for j,bvec in enumerate(arraylist):
#               
#             dtemp = Eu_distance(avec,bvec)
#             distance_matrix[i][j] = dtemp
#             sumd +=dtemp
    aveSum = sumd /(size*size)

    return distance_matrix
def precomputing_distance_pre(arraylist,prelist):
  
  
    size_w = len(arraylist)
    size_h = len(prelist)
    distance_matrix = zeros((size_h+size_w, size_h+size_w), float)
    temp = list(arraylist)
    for item in prelist:
        temp.append(item)
    sim_matrix = cosine_similarity(array(temp))
    i = 0
    for i in range(size_w+size_h):
        for j in range(size_w+size_h):
            #print sim_matrix[i][j]
            distance_matrix[i][j] = distance_matrix[j][i] = 1 - sim_matrix[i][j]
    return array(distance_matrix)

if __name__ == "__main__":
    a = [[1, 2,1], [2, 1,0]]
    print precomputing_distance(a)
