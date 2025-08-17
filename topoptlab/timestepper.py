from typing import Tuple

import numpy as np
from scipy.sparse import csc_array

def bdf_coefficients(k: int) -> np.ndarray:
    """
    Get coefficients for backward differentiation formula BDF. Shamelessly 
    copied from Elmer Solver manual. Also check the formula there to 
    understand what the coefficients mean.

    Parameters
    ----------
    k : int
        order of BD. must be equal or smaller 6.

    Returns
    -------
    coefficients : np.ndarray shape (k+1)
        First one is multiplied with the forces and the stiffness matrix, the 
        others on the right hand side with the mass matrix and the history of 
        the function.
    """
    if k > 6:
        raise NotImplementedError("Not implemented for order higher than 6.")
    return np.array([[1,1],
                     [2/3,4/3,-1/3],
                     [6/11,18/11,-9/11,2/11],
                     [12/25,48/25,-36/25,16/25,-3/25],
                     [60/137,300/137,-300/137,200/137,-75/137,12/137],
                     [60/147,360/147,-450/147,400/147,-225/147,72/147,-10/147]])[k-1]

def backward_diff(M: csc_array, K: csc_array,
                  f: np.ndarray, phi: np.ndarray, 
                  h: float, order: int) -> Tuple[csc_array,np.ndarray]:
    """
    Return left hand side (matrix) and right hand side for one timestep 
    evolution of the backward difference. We assume the generic PDE of first 
    order in time 
    
    d phi / dt + D phi = f
    
    where phi is the evolution variable (what we want to solve for), D a 
    differential operator purely in space (not time) and f a generic function 
    of space and time. After discretization in space, we convert phi to its 
    discretized equivalent Phi and the equation becomes
    
    M d phi / dt + K Phi = f
    
    with the discretization of the identity operator M (often called mass 
    matrix) and matrix of the discretization of the operator D which we call K.
    
    Parameters
    ----------
    M : csc_array
        discretization of identity (mass matrix).
    K : csc_array
        discretization of operator D (e. g. stiffness matrix).
    f : np.ndarray
        generic right hand side function.
    phi : np.ndarray
        variable to solve for.
    h : int
        time step width.
    order : int
        order of BD. must be equal or smaller 6.

    Returns
    -------
    lhs : csc_array
        First one is multiplied with the forces and the stiffness matrix, the 
        others on the right hand side with the mass matrix and the history of 
        the function.
    rhs : np.ndarray
        vector of the right hand side
    """
    coeffs = bdf_coefficients(k=order)
    return M/h + coeffs[0]*K, coeffs[0]*f + M/h @ (coeffs[1:] * phi[-1:-order-1:-1]).sum(axis=1) 