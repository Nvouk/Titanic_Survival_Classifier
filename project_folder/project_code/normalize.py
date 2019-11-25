from numpy import *
from compute_age import compute_age

def normalize(X):
    
    # afairesh twn id
    X = delete(X, 0, 1)
    
    # afairesh twn tickets
    X = delete(X, 6, 1)
    
    # fix names    
    for i in range(X.shape[0]):
        name = X[i,1]
        init = name.find(',')
        end = name.find('.')
        name = name[init+2:end]
        
        if name=='Mrs':
            X[i,1] = 1
        elif name=='Miss':
            X[i,1]=2
        elif name=='Master':
            X[i,1]=3
        elif name=='Mr':
            X[i,1]=6
        else:
            X[i,1]=4
                

    # fix gender
    for i in range(X.shape[0]):
        gender = X[i,2]
        if gender == 'male':
            X[i,2] = 1
        else:
            X[i,2] = -1
            
    # fix cabins
    for i in range(X.shape[0]):
        cabin = X[i,7]
        cabin = cabin[0:1]
        
        # cabin class
        if (cabin == 'A') or (cabin == 'B') or (cabin == 'C') or (cabin == 'D') or (cabin == 'T'):
            X[i,7] = 1
        elif cabin == 'E':
            X[i,7] = 2
        elif (cabin == 'F') or (cabin == 'G'):
            X[i,7] = 3
        else:
            X[i,7] = 0
        
    # fix embarkment port
    for i in range(X.shape[0]):
        port = X[i,8]
        if port == 'S':
            X[i,8] = 1
        elif port == 'C':
            X[i,8] = 2
        else:
            X[i,8] = 3
            
    # newX : neos pinakas pou tha pairnei kathe stili tou X(string)
    #        kai tha tin metatrepei se arithmo (int)
    newX = [ [ 0 for i in range(X.shape[0]) ] for j in range(X.shape[1]) ]
    
    # Diatrexoume ton X kathe grammi tou
    for i in range(X.shape[0]):
        # ... kai gia tis 3 prwtes stiles --> (int)
        for j in range(3):
            newX[j][i] = int(X[i,j])
        
        # gia tis stiles 5,6 --> (int)
        for k in range(4,6):
            newX[k][i] = int(X[i,k])
        
        # gia tis stiles 8,9 --> (int)
        for m in range(7,9):
            newX[m][i] = int(X[i,m])
        
        # an i 4i stili(age) exei arithmo float...
        if isfloat(X[i,3]):
            # ...stroggilopoihsh prwta to float kai meta se int
            newX[3][i] = int(round(float(X[i,3])))
        
        # an i 7i stili(fare) exei arithmo float...
        if isfloat(X[i,6]):
            # ...stroggilopoihsh prwta to float kai meta se int
            newX[6][i] = int(round(float(X[i,6])))
            
    # transpose ton pinaka gia na ginoun oi grammes stiles
    # etsi wste na ton exoume stin morfi pou theloume
    newX=transpose(newX)
    
    # ypologismos tis kenis-midenikis ilikias
    # options:
    # 1: kNN, 2: m.o olwn , 3: m.o katigoriwn onomatos , 4: m.o gender
    newX = compute_age(newX,2)
    
    # ipologismos plithous oikogeneias
    family = add(newX[:,4], newX[:,5])
    newX = column_stack([newX, family])   
    
    #afairesi sibsp kai parch
    newX = delete(newX,5,1)
    newX = delete(newX,4,1)
    
    #ipologismos mesou orou timhs eisithriou
    mean = 0
    sum = 0
    min = 99999999999
    max = 0
    for i in range(newX.shape[0]):
        sum = sum + newX[i,4]
        if newX[i,4] < min:
            min = newX[i,4]
        if newX[i,4] > max:
            max = newX[i,4]
            
    mean = sum/newX.shape[0]

    #mesh timh metaksi mesou kai min
    mean1 = (mean+min)/2
    #mesh timh metaksi mesou kai max
    mean2 = (mean+max)/2
    
    #sthlh me kathgoriopoihsh timwn ana pclass
    money = zeros((newX.shape[0], 1))
    
    for i in range(newX.shape[0]):
        pclass = newX[i, 0]
        fare = newX[i, 4]
        if pclass == 1:
            if fare >= min and fare < mean1:
                money[i] = 7
            elif fare >= mean1 and fare < mean2:
                money[i] = 4
            elif fare >= mean2 and fare <= max:
                money[i] = 1
                
        elif pclass == 2:
            if fare >= min and fare < mean1:
                money[i] = 8
            elif fare >= mean1 and fare < mean2:
                money[i] = 5
            elif fare >= mean2 and fare <= max:
                money[i] = 2
                
        elif pclass == 3:
            if fare >= min and fare < mean1:
                money[i] = 9
            elif fare >= mean1 and fare < mean2:
                money[i] = 6
            elif fare >= mean2 and fare <= max:
                money[i] = 3
                
    newX = column_stack([newX, money])

    #norma ilikiwn
    norma = linalg.norm(newX[:,3])
    
    #sthlh me kathgoriopoihsh ilikias ana titlo kai fylo
    newAge = zeros((newX.shape[0], 1))
    for i in range(newX.shape[0]):
        title = newX[i, 1]
        gender = newX[i, 2]
        age = newX[i, 3]
        normAge = ((title*age*gender)*100)/norma
        normAge = round(normAge)
        newAge[i] = normAge
        
    newX = column_stack([newX, newAge])
    
    #norma timwn
    norma = linalg.norm(newX[:,4])
    
    #sthlh me kathgoriopoihsh timhs ana limani kai cabin
    port = zeros((newX.shape[0], 1))
    for i in range(newX.shape[0]):
        fare = newX[i, 4]
        embark = newX[i, 6]
        cabin = newX[i, 5]
        normPort = ((fare*embark*cabin)*100)/norma
        normPort = round(normPort)
        port[i] = normPort
        
    newX = column_stack([newX, port])
    
    #norma family
    norma = linalg.norm(newX[:,7])
    
    #sthlh me kathgoriopoihsh family ana titlo
    status = zeros((newX.shape[0], 1))
    for i in range(newX.shape[0]):
        family = newX[i, 7]
        title = newX[i, 1]
        gender = newX[i, 2]
        normStatus = ((family*title*gender)*100)/norma
        normStatus = round(normStatus)
        status[i] = normStatus
        
    newX = column_stack([newX, status])
    
    return newX

# methos pou elegxei an mia timi (string) einai float
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False