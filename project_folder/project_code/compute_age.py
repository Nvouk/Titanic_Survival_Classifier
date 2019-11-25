from kNN.euclideanDistance import euclideanDistance
from numpy import *

# methodos pou ypologizei tis midenikes times twn ilikiwn pou emfanizontai
# tis vriskei me 4 methodous ( choice )
# 1: kNN, 2: m.o olwn , 3: m.o katigorias onomatos , 4: m.o gender

def compute_age(X,choice):
    
    # olokliri i stili twn ilikiwn
    col_age = X[:,3]
    
    # olokliri i stili tis katigorias tou onomatos
    col_name = X[:,1]
    
    # olokliri i stili tou gender
    col_gender = X[:,2]
    
    # mesos oros
    mean = 0
    
    # to index(grammi) kathe ilikias
    index = 0
       
    # Ypologismos tis midenikis ilikias me kNN
    #### !!!! NOTE !!! #####
    # O kNN den ginetai na doulepsei toso swsta gt pairnei tis pio kontines apostaseis
    # oi opoioi einai autoi pou exoun ilikia konta sto 0 h 0. Ara trexontas ton algorithmo
    # bgainoun times polu mikres s autous pou prin eixan timi 0. Ara oxi kali methodos(argi episis)
    
    if choice == 1:
        
        # k neighboors
        k = 30 

        # pinakas xoris tis ilikies
        A = delete(X,3,1)
        
        # gia kathe ilikia stin stili
        for age in col_age:
            # an kapou yparxei i timi 0...
            
            if age == 0:
            
                distance=[]
                # ...ypologise tin apostasi autis tis grammis me oles tis ypoloipes
                for i in range(A.shape[0]):
                    distance.append(euclideanDistance(A[i,:],A[index,:]))
                
                # apothikeuse tis apostaseis kathws kai tis antistoixes ilikies
                sortedDistances,sortedAges = zip(*sorted(zip(distance,col_age)))
                
                # kratame tis k kontinoteres ilikies pou den einai 0
                # giati alliws oi polloi kontinoi exoun ilikia 0
                # diladi pou den exei ypologistei akoma kai to lamvanoume ws
                # false positive (alla kai pali den einai veltisto)
                kAges = []
                for j in sortedAges:
                    if j != 0:
                        kAges.append(j)
                
                kAges = kAges[0:k]
                
                # ypologizoume ton meso oro twn k kontinoterwn ilikiwn
                s = 0;
                for j in kAges:
                    s += j
                
                mean = s / k
                
                # orizoume stin sugkekrimeni thesi ton meso oro pou vrikame
                X[index,3] = mean
                
                #print X[index,3]
            index += 1        
        
    # ypologismos tis ilikias me meso oro olwn twn ilikiwn            
    elif choice == 2:
        # gia ola tis times twn age
        for age_zero in col_age:
            # an einai 0 i ilikia...
            if age_zero == 0:
                # sum
                s = 0
                # arithmos emfanisewn
                count = 0
                # ...diatrexoume pali oles tis mi midenikes ilikies
                # kai ypologizoume ton meso oro autwn
                for age_normal in col_age:
                    if age_normal != 0:
                        count += 1
                        s += age_normal            
                
                mean = s / count
                # orizoume stin sugkekrimeni thesi ton meso oro pou vrikame
                X[index,3] = mean
            index += 1
                
    
    # ypologismos tis ilikias me meso oro twn antistoixwn katigoriwn onomatwn
    # px Mr,Mss,Master,etc
    elif choice == 3:
        # gia oles tis ilikies pou einai miden...
        for age_zero in col_age:
            if age_zero == 0:
                # arithmos emfanisewn
                count = 0
                # index of name
                index_n = 0
                # sum                
                s = 0
                # Mr,Mss,etc gia ti sugkekrimeni ilikia
                name_cat = X[index,1]
                
                # gia to onoma pou vrikes prin, diatre3e ti stili me ta onomata
                # kai kathe fora pou vriskeis ena au3ise ton counter
                # kai ypologise to athroisma twn ilikiwn autwn kai meta ton meso oro autwn
                for name_cat in col_name:
                    count += 1
                    s += X[index_n,3]
                
                mean = s / count
                
                # orizoume stin sugkekrimeni thesi ton meso oro pou vrikame
                X[index,3] = mean
                    
            index += 1
            
            
    # ypologismos tis ilikias me meso oro twn ilikiwn ana gender        
    elif choice == 4:
        # gia oles tis ilikies pou einai miden...
        for age_zero in col_age:
            if age_zero == 0:
                # arithmos emfanisewn
                count = 0
                # index of gender
                index_g = 0
                # sum                
                s = 0
                # gender gia ti sugkekrimeni ilikia
                gender = X[index,1]
                
                # gia to gender pou vrikes prin, diatre3e ti stili me ta gender
                # kai kathe fora pou vriskeis ena au3ise ton counter
                # kai ypologise to athroisma twn ilikiwn autwn kai meta ton meso oro autwn
                for gender in col_gender:
                    count += 1
                    s += X[index_g,3]
                
                mean = s / count
                
                # orizoume stin sugkekrimeni thesi ton meso oro pou vrikame
                X[index,3] = mean
                    
            index += 1


    return X
            