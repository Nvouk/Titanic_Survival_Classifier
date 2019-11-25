from numpy import *
from euclideanDistance import euclideanDistance

def kNN(k, X, labels, y):
# Assigns to the test instance the label of the majority of the labels of the k closest 
# training examples using the kNN with euclidean distance.

    n = X.shape[0] # number of training examples (rows)
    m = X.shape[1] # number of attributes (columns)
    p = y.shape[0] # number of testing examples (rows)
    
    #closest = zeros((k,2)) # stores the k closest to the test instance training examples and the distances
    
    # array to store the predicted Labels
    predictedLabels = zeros(p)    
    
    #calculate and save distance between y and every training example  
    for i in range (p):
        #array that stores y distances from training set examples
        distance=[]
        for j in range(n):
            distance.append(euclideanDistance(X[j,:],y[i,:]))
        
        #sort distances-labels
        sortedDistances, sortedLabels = zip(*sorted(zip(distance,labels)))
        
        #print sortedDistances
    
        #get K nearest examples
        #kDistances = sortedDistances[0:k]
        kLabels = sortedLabels[0:k]
        
        # counter for getting the number of 0 or 1
        counter = zeros(2)
        
        #decide label
        for m in range (k):
            if kLabels[m]==0:
                counter[0]+=1
            elif kLabels[m]==1:
                counter[1]+=1
                
        if counter[0] > counter[1]:
            predictedLabels[i]=0
        else:
            predictedLabels[i]=1
            
    #print predictedLabels

    return predictedLabels
