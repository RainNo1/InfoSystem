#-*- coding:utf8 -*-

def vector_multiplier(vector1,vector2):
    result = 0.0
    if len(vector1) == len(vector2):
        for i in range(len(vector2)):
            result += vector1[i]*vector2[i]
    return result


# vector addiction operation     
def vector_blend(vec1,vec2):
    if not len(vec1)==len(vec2):
        return None
    for i in range(len(vec1)):
        vec1[i] = vec1[i]+vec2[i]
    return vec1

def vectors_blend(vectors, weights):
    n = len(vectors)
    if n > 0: 
        if n == 1:
            return vectors[0]
        dim = len(vectors[0])
        vec = [0.0 for d in range(dim)]
                
        for i in range(n):
            for d in range(dim):
                vec[d] += vectors[i][d] * weights[i]
        return vec
    return None
if __name__=="__main__":
    print "start..."
    