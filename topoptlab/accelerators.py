# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,List,Union
from warnings import warn

import numpy as np
from scipy.linalg import lstsq

def _quasinewton_pairs(x: np.ndarray,
                       xhist: List[np.ndarray],
                        g: np.ndarray,
                       ghist: List[np.ndarray],
                       max_history: int
                       ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Build quasi-Newton curvature pairs from a finite history of iterates and
    gradients given the current iterate x_[k] and gradient g_[k] together
    with their histories. Build the difference pairs

        s_[i] = x_[i+1] - x_[i]
        y_[i] = g_[i+1] - g_[i]

    are assembled into the matrices

        S = [s_[k-m], ..., s_[k-1]]
        Y = [y_[k-m], ..., y_[k-1]]

    where m = min(k, max_history). The pairs satisfy the secant condition

        Y[:, i] ≈ H @ S[:, i]

    and ca be used to build quasi-Newton approximations of the Hessian or 
    its inverse, e.g. in L-BFGS or SR1 updates.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        current iterate x_[k].
    xhist : list of np.ndarray
        history of past iterates. The last element is x_[k-1].
    g : np.ndarray, shape (n,)
        current gradient g_[k].
    ghist : list of np.ndarray
        history of past gradients. The last element is g_[k-1].
    max_history : int
        maximum number of pairs to retain.

    Returns
    -------
    S : np.ndarray, shape (n, m)
        difference matrix.
    Y : np.ndarray, shape (n, m)
        Gradient difference matrix.
    """
    if len(xhist) < 1 or len(ghist) < 1:
        raise ValueError("Need at least one past iterate and gradient.")
    #
    X = np.column_stack(xhist[-max_history:] + [x])
    G = np.column_stack(ghist[-max_history:] + [g])
    #
    return X[:, 1:] - X[:, :-1], G[:, 1:] - G[:, :-1]

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
    
    We assume that we have the current iterate x_[k+1] available and a history 
    of previous iterates as well. In particular, the last entry of the history 
    is x_[k]. We define the incremental matrix
    
        dX = [x_[k-m+1] - x_[k-m], ... , x_[k] - x_[k-1]],
    
    the residual
    
        r_[i] = x_[i+1] - x_[i]
    
    and the incremental residual matrix
    
        dR = [r_[k-m+1] - r_[k-m], ... , r_[k] - r_[k-1]].
    
    To accelerate we find the gamma that minimizes 
    
        ||dR@gamma - r_[k]||_[2]
    
    and find the updated x_[k+1]
    
        x_[k+1] = x_[k] + damp*r_[k] - (dX + damp*dR)@gamma
    
    where damp is a damping parameter between zero and one. For details check the 
    wikipedia article or 
    
    Pratapa, Phanisri P., Phanish Suryanarayana, and John E. Pask. "Anderson 
    acceleration of the Jacobi iterative method: An efficient alternative to 
    Krylov methods for large, sparse linear systems." Journal of Computational 
    Physics 306 (2016): 43-54.
    
    Parameters
    ----------
    x : np.ndarray (n)
        current fixed point iterate x_[k+1] = f(x_[k])
    xhist : list
        history of iterations. The last element of the list is x_[k].
    max_history : int
        maximum number of past results used for the current update.
    damp : float
        damping applied to Anderson update.

    Returns
    -------
    x : np.ndarray
        updated iterate.
    """
    if len(xhist) == 0: 
        raise ValueError("xhist is empty.")
    elif max_history <= 1:
        raise ValueError("max_history <= 1 cannot be used for a sensible Anderson acceleration.")
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
