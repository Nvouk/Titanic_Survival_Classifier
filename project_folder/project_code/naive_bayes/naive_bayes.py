from numpy import *

def naive_bayes(trainSet,trainLabels,testSet):
    
     predictedLabels = zeros(testSet.shape[0])
     features = {}
     survived = 0
     drowned = 0
     
     for i in range(trainSet.shape[0]):
         if trainLabels[i] == 1:
             survived += 1
             for j in range(trainSet.shape[1]):
                 if trainSet[i,j] in features:
                     features[trainSet[i,j]]['survived'] += 1
                 else:
                     features[trainSet[i,j]] = {'survived':1, 'drowned':0}
                
         else:
             drowned += 1
             for j in range(trainSet.shape[1]):
                 if trainSet[i,j] in features:
                     features[trainSet[i,j]]['drowned'] += 1
                 else:
                     features[trainSet[i,j]] = {'survived':0, 'drowned':1}
        
        
    
     totalPassengers = survived + drowned
     prSurvived = survived/float(totalPassengers)
     prDrowned = drowned/float(totalPassengers)
     
     pSurvived = prSurvived
     pDrowned = prDrowned
     
     for i in range(testSet.shape[0]):
         for j in range(testSet.shape[1]):
             if testSet[i,j] in features:
                 pSurvived *= features[testSet[i,j]]['survived']
                 
                 pDrowned *= features[testSet[i,j]]['drowned']
        
         if pSurvived > pDrowned:
             predictedLabels[i] = 1
             
             
     return predictedLabels;