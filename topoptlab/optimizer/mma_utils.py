# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Dict,Union

import numpy as np

from mmapy import mmasub

def mma_defaultkws(n: int, n_constr: int, ft: Union[None,int] = None) -> Dict:
    """
    Default arguments for the method of moving asymptotes (MMA).
    
    Parameters
    ----------
    n : int
        number of design variables.
    n_constr : int
        number of constraints.
    ft : None or int
        filter flag. If None, adopts default move limit of 0.2.
        
    Returns
    -------
    optimizer_kw : dict
        dictionary filled with default arguments.

    """
    if n_constr == 0:
        raise ValueError("GCMMA needs to have at least one constraint.")
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
    #
    optimizer_kw["asyinit"] = 0.5
    optimizer_kw["asydecr"] = 0.7
    optimizer_kw["asyincr"] = 1.2
    optimizer_kw["asymin"] = 0.01
    optimizer_kw["asymax"] = 10
    optimizer_kw["raa0"] = 1e-5
    optimizer_kw["albefa"] = 0.1
    return optimizer_kw

def gcmma_defaultkws(n: int, ft: int, n_constr: int) -> Dict:
    """
    Default arguments for the globally convergent method of moving asymptotes 
    (GCMMA).
    
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
    if n_constr == 0:
        raise ValueError("GCMMA needs to have at least one constraint.")
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
    #
    optimizer_kw["asyinit"] = 0.5
    optimizer_kw["asydecr"] = 0.7
    optimizer_kw["asyincr"] = 1.2
    optimizer_kw["asymin"] = 0.01
    optimizer_kw["asymax"] = 10
    optimizer_kw["epsimin"] = 1e-7
    optimizer_kw["raa0"] = 1e-2
    optimizer_kw["raa"] = 0.01*np.ones((n_constr,1))
    optimizer_kw["raa0eps"] = 1e-6
    optimizer_kw["raaeps"] = 1e-6*np.ones((n_constr,1))
    optimizer_kw["albefa"] = 0.1
    return optimizer_kw

def update_mma(x,xold1,xold2,xPhys,
               obj,dobj,constrs,dconstr,
               iteration,
               nconstr,xmin,xmax,low,upp,
               a0,a,c,d,move,
               **kwargs):
    """
    This function is purely legacy and will be removed in future versions.
    """
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj 
    df0dx = mu0*dobj[None].T
    xval = x.copy()[None].T 
    #
    fval=mu1*constrs[:,None]
    dfdx=mu1*np.atleast_2d(dconstr.T)
    return mmasub(m=nconstr,
                  n=x.shape[0],
                  iter=iteration,
                  xval=xval,
                  xmin=xmin,
                  xmax=xmax,
                  xold1=xold1,
                  xold2=xold2,
                  f0val=f0val,
                  df0dx=df0dx,
                  fval=fval,
                  dfdx=dfdx,
                  low=low,
                  upp=upp,
                  a0=a0,
                  a=a,
                  c=c,
                  d=d,
                  move=move)