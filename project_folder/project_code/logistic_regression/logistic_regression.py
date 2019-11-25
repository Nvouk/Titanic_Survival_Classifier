from numpy import *
import scipy.optimize as op
from computeCost import computeCost
from computeGrad import computeGrad
from predict import predict

def logistic_regression(trainSet,trainLabels,testSet): 
     
    #Add intercept term to X
    X_new = ones((trainSet.shape[0], trainSet.shape[1]+1))    
    X_new[:, 1:trainSet.shape[1]+1] = trainSet
    trainSet = X_new
    
	#Add intercept term to X
    X_new = ones((testSet.shape[0], testSet.shape[1]+1))    
    X_new[:, 1:testSet.shape[1]+1] = testSet
    testSet = X_new
    
    #find the number of attributes
    numberOfAttributes=trainSet.shape[1]   
    
    #initialize theta
    temp_theta=zeros((numberOfAttributes,1))
    
    # Minimize cost
    Result = op.minimize(fun = computeCost, x0 = temp_theta, args = (trainSet, trainLabels), method = 'TNC',jac = computeGrad);
    # Save results
    theta = Result.x;
    
    # Predict labels of testSet
    predictedLabels = predict(array(theta), testSet)
   # predictedLabels= zeros(testSet.shape[0])
    return predictedLabels;