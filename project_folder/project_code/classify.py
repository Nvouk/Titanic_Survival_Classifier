from numpy import *
from kNN.kNN import kNN
from logistic_regression.logistic_regression import logistic_regression
from naive_bayes.naive_bayes import naive_bayes

def classify(trainSet, trainLabels, testSet):
    
    # Logistic Regression
    predictedLabels=logistic_regression(trainSet, trainLabels, testSet)

    # kNN
    #predictedLabels = kNN(10,trainSet,trainLabels,testSet)
    
    # Naive Bayes
    #predictedLabels = naive_bayes(trainSet,trainLabels,testSet)
    
    return predictedLabels
