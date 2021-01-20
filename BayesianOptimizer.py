from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
from EI import EI
import numpy as np
from sklearn.linear_model import LinearRegression as LM


class BayesianOptimizer:
    #wDomain and bDomain are arrays: [min,max,increment]
    def __init__(self, wDomain, bDomain):
        wDomain = np.arange(wDomain[0],wDomain[1], wDomain[2])
        bDomain = np.arange(bDomain[0],bDomain[1], bDomain[2])
        self.GPRAccuracys = []
        self.GPRStd = []
        self.domain = []
        for w in wDomain:
            for b in bDomain:
                self.domain.append([w,b])
    @staticmethod
    def modelAcc(x,y,w,b):
        sqErr = sum([((w*x[i]+b)-y[i])**2 for i in range(len(x))])
        nMSE = -sqErr/len(x)
        return nMSE
    
    def GPRDistribution(self,params, scores):           
        gpr = GPR(kernel=RBF(),random_state=0).fit(params, scores)
        predAcc,std = gpr.predict(self.domain, return_std=True)
        self.GPRAccuracys.append(predAcc)
        self.GPRStd.append(std)
        return [predAcc,std]
        
        
        
    def fit(self, x,y, thresh):
        w = 5
        b = 5
        scores = []
        params = []
        inARow = 0
        while(inARow<thresh):
            curAcc = self.modelAcc(x,y,w,b)
            if(len(params)!=0 and params[-1]==[w,b]):
                inARow+=1
            else:
                inARow = 0
            scores.append(curAcc)
            params.append([w,b])
            
            predAccDist,std = self.GPRDistribution(params, scores)
            w,b = EI.MaxExpectedImprovement(self.domain, predAccDist,std)
        print("Final weights are: ", w, b)
        self.paramHist = params
        self.scoreHist = scores
        
            
            
            
            


import numpy as np
import matplotlib.pyplot as plt
from BayesianOptimizer import BayesianOptimizer as BO


x = [x for x in range(20)]
y = [2*x+4 for x in range(20)]
Bop = BO([0,10,0.1],[0,10,0.1])
Bop.fit(x,y, 10)
    
    
