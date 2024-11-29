import numpy as np

from mmapy import mmasub

def update_mma(x,xold1,xold2,xPhys,
               obj,dc,constrs,dconstr,
               iteration,
               nconstr,xmin,xmax,low,upp,
               a0,a,c,d,move):
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj 
    df0dx = mu0*dc[np.newaxis].T
    #fval = mu1*np.array([[xPhys.mean()-volfrac]])
    #dfdx = mu1*(dconstr/(x.shape[0]*volfrac))
    xval = x.copy()[np.newaxis].T 
    #print(obj.shape, f0val.shape)
    #print(df0dx.shape)
    #print(constrs.shape)
    #print(dconstr.T.shape)
    #print(xval.shape)
    #raise ValueError()
    return mmasub(nconstr,x.shape[0],iteration,
                  xval,xmin,xmax,
                  xold1,xold2,f0val,df0dx,
                  mu1*constrs[:,None],mu1*np.atleast_2d(dconstr.T),
                  low,upp,
                  a0,a,c,d,move)