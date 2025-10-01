# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,List,Union
from warnings import warn

import numpy as np
from scipy.linalg import lstsq

def anderson(x: np.ndarray, 
             xhist: List, max_history: int,
             damp: float = 0.9,
             **kwargs: Any) -> np.ndarray:
    """
    Anderson acceleration to achieve convergence acceleration. It assumes 
    that the numerical process resembles a fixed point iteration
    
    x_[i+1] = f(x_[i])
    
    that we seek to accelerate to achieve the solution 
    
    f(x)-x = 0.
    
    We assume that we have the current iterate x_[i+1] available and a history 
    of q+1 iterates as well. We define the incremental matrix
    
    dX = [x_[k-m+1] - x_[k-m], ... , x_[k] - x_[k-1]],
    
    the residual
    
    r_[i] = x_[i+1] - x[i]
    
    and the incremental residual matrix
    
    dR = [r_[k-m] ... r_[k]]
    
    To accelerate we find the gamma that minimizes 
    
    ||dR@gamma - r_[k]||_[2]
    
    and find the updated x_[k+1]
    
    x_[k+1] = x_[k+1] - (dX + damp*dRx)@gamma
    
    where damp is a damping parameter between zero and one. 
    
    For details check the wikipedia article or 
    
    Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. "Anderson 
    acceleration of the Jacobi iterative method: An efficient alternative to 
    Krylov methods for large, sparse linear systems." Journal of Computational 
    Physics 306 (2016): 43-54.
    
    Parameters
    ----------
    x : np.ndarray (n)
        current iterate
    xhist : list
        history of iterations. The last element of the list
    max_history : int
        maximum number of past results used for the current update.
    damp : float
        damping applied to Anderson update.

    Returns
    -------
    x : np.ndarray
        updated iterate.
    """
    # assemble to adequate matrix
    X = np.column_stack(xhist[-max_history:]+[x])
    R = X[:,1:] - X[:,:-1]
    # differences of x and residuals
    dX = X[:,1:-1] - X[:,:-2]
    dR = R[:,1:] - R[:,:-1]
    # solve for coefficients gamma
    gamma,res,rank,s = lstsq(dR,R[:,-1])
    x = xhist[-1]*(1-damp) + x*damp - (dX+damp*dR)@gamma
    return x

def diis(x: np.ndarray, xhist: List, 
         max_history: int,
         r: Union[None,np.ndarray] = None, 
         rhist: Union[None,List] = None,
         damp: float = 0.9) -> np.ndarray:
    """
    Direct inversion in the iterative subspace (DIIS) or also known as Pulay 
    mixing for convergence acceleration. Two use cases have to be 
    distinguished: i) a residual is available (e. g. we try to solve a linear 
    system iteratively r = b - A@x) ii) no residual is available meaning we 
    perform a recursion (e. g. optimization or a fixed point iteration). 
    
    Parameters
    ----------
    x : np.ndarray (n)
        current iterate
    xhist : list
        history of iterations. The current iterate is not in this list.
    max_history : int
        maximum number of past results used for the current update.
    r : np.ndarray (n)
        current residual (e. g. from a linear system ala r=b-A<qx)
    rhist : list
        history of residuals.
    damp : float
        damping applied to DIIS update.

    Returns
    -------
    x : np.ndarray
        updated iterate.
    """ 
    warn("Currently not tested and might still contain bugs.")
    n = len(xhist)
    if n < 2 and rhist is not None:
        raise ValueError("Need at least two past result for DIIS acceleration.")
    X = np.column_stack(xhist[-max_history:]+[x])
    # calculate residuals
    if r is None and rhist is None:
        R = X[:,1:] - X[:,:-1]
    elif r is not None and rhist is None:
        R = X[:,1:] - X[:,:-1]
    else:
        rhist = rhist + [r]
        R = np.column_stack(rhist)
        n = n+1
    #
    #print("R unnormalized",R,"\n")
    #norm = np.linalg.norm(R,2,axis=0)
    #R = R / norm
    #print("R normalized",R,"\n")
    # build B matrix: B_ij = <r_i | r_j>
    B = np.zeros((n, n))
    i=0
    # off-diagonal
    for i in np.arange(R.shape[1]-1):
        B[i,i+1:-1] = R[:,i].dot(R[:,i+1:])
    #print(B,"\n")
    B = B + B.T
    #print(B,"\n")
    # diagonal
    B = B + np.eye(R.shape[1]+1)
    #print(B,"\n")
    #
    B[-1, :-1] = -1
    B[:-1, -1] = -1
    B[-1, -1] = 0
    # 
    rhs = np.zeros(n)
    rhs[-1] = -1
    try:
        coeffs = np.linalg.solve(B, rhs)[:-1]
    except np.linalg.LinAlgError as err:
        print(X)
        print(R)
        print(B)
        print(rhs)
        raise np.linalg.LinAlgError(err)
    #print(B,"\n")
    # update
    x = xhist[-1]*(1-damp) + damp*np.column_stack([c*_x for c,_x in zip(coeffs,xhist) ]).sum(axis=1)
    return x
