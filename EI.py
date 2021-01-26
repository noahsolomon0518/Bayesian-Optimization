import numpy as np
def MaxExpectedImprovement(x,y,std):
    maxImprovement = 0
    improvement = y+3.66*std
    maxScore = max(improvement)
    argmax = np.argmax(improvement)



    return x[argmax]