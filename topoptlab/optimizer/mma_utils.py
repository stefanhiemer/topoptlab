import numpy as np

from mmapy import mmasub

def mma_defaultkws(n,ft,n_constr):
    """
    Default arguments for the method of moving asymptotes (MMA) based on 
    experience.
    
    Parameters
    ----------
    n : int
        number of design variables.
    ft : int
        filter flag.
    n_constr : int
        number of constraints.
        
    Returns
    -------
    optimizer_kw : dict
        dictionary filled with default arguments.

    """
    #
    optimizer_kw = {}
    # number of constraints
    optimizer_kw["nconstr"] = n_constr
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.zeros((n,1))
    optimizer_kw["xmax"] = np.ones((n,1))
    # initial lower and upper asymptotes
    optimizer_kw["low"] = np.ones((n,1))
    optimizer_kw["upp"] = np.ones((n,1))
    #
    optimizer_kw["a0"] = 1.0
    optimizer_kw["a"] = np.zeros((n_constr,1))
    optimizer_kw["c"] = 10000*np.ones((n_constr,1))
    optimizer_kw["d"] = np.zeros((n_constr,1))
    #
    if ft in [5,6]: 
        optimizer_kw["move"] = 0.1
    elif ft in [7]:
        optimizer_kw["move"] = 0.05
    else:
        optimizer_kw["move"] = 0.2
    return optimizer_kw

def update_mma(x,xold1,xold2,xPhys,
               obj,dobj,constrs,dconstr,
               iteration,
               nconstr,xmin,xmax,low,upp,
               a0,a,c,d,move,
               **kwargs):
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj 
    df0dx = mu0*dobj[None].T
    xval = x.copy()[None].T 
    return mmasub(nconstr,x.shape[0],iteration,
                  xval,xmin,xmax,
                  xold1,xold2,f0val,df0dx,
                  mu1*constrs[:,None],mu1*np.atleast_2d(dconstr.T),
                  low,upp,
                  a0,a,c,d,move)