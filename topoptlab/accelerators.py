import numpy as np
from scipy.linalg import lstsq

def anderson(x,xhist,max_history,
             damp=0.9):
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

def diis(x, xhist, 
         max_history,
         r=None, rhist=None,
         damp=0.9):
    """
    Direct inversion in the iterative subspace (DIIS) or Pulay mixing.
    
    Parameters
    ----------
    x : np.ndarray (n)
        current iterate
    xhist : list
        history of iterations. The current iterate is not in this list.
    max_history : int
        maximum number of past results used for the current update.
    r : np.ndarray (n)
        current residual (e. g. from a linear system ala r=b-Ax)
    rhist : list
        history of residuals.
    damp : float
        damping applied to DIIS update.

    Returns
    -------
    x : np.ndarray
        updated iterate.
    """ 
    n = len(xhist)
    if n < 1:
        raise ValueError("Need at least 1 past result for diis acceleration.")
    X = np.column_stack(xhist[-max_history:]+[x])
    # calculate residuals
    if r is None and rhist is None:
        rhist = X[:,1:] - X[:,:-1]
    # Build B matrix: B_ij = <r_i | r_j>
    B = np.empty((n + 1, n + 1))
    for i in np.arange(n):
        B[i,i:-1] = rhist[:,i].dot(rhist[:,i:])
        
    #B[:-1, :-1] = np.array([[np.dot(rhist[i], rhist[j]) for j in range(n)] \
    #                         for i in range(n)])
    B[-1, :-1] = -1
    B[:-1, -1] = -1
    B[-1, -1] = 0
    # 
    rhs = np.zeros(n+1)
    rhs[-1] = -1
    coeffs = np.linalg.solve(B, rhs)[:-1]
    # update
    x = xhist[-1]*(1-damp) + damp*np.column_stack([c*_x for c,_x in zip(coeffs,xhist) ]).sum(axis=1)
    return x
