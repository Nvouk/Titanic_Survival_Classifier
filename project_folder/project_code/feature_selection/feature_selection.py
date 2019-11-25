from numpy import *
from chiSQ import chiSQ
from infogain import infogain

# feature selection gia ton pinaka X kratontas k xaraktiristika
def feature_selection(X,k,s,Y):

    # arithmos ton features pou kratame
    num_feat = k

    # epilogoume tin methodo epilogis xarattiristikon
    if s==1:
        gain = infogain(X,Y)
    elif s==2: 
        gain = chiSQ(X,Y)

    
    index = argsort(gain)[::-1]

    # select the top num_feat features
    X = X[:,index[:num_feat]]
    
    return X