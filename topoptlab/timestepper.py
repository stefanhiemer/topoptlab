# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Tuple, Union

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
        mass matrix.
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

def bossak_init(alpha: float = -0.05, **kwargs: Any) -> Tuple[float,float]:
    """
    Calculate parameters beta and gamma for the Bossak method:
        
    beta = 1/4 * (1-alpha)**2
    gamma = 1/2 - alpha
    
    Parameters
    ----------
    alpha : float
        Bossak parameter typically between -0.3 and 0. The more negative, the 
        more damping is done. 0 recovers the Newmark beta method with middle 
        point rule if equal to zero (unconditionally stable in lin. problems).
        
    Returns
    -------
    beta : float
        mixing parameter that determines how much of the new acceleration 
        enters the displacement update.
    gamma : float
        mixing parameter for velocity update that determines how much of the 
        new acceleration enters the velocity update.
    """
    
    return 1/4 * (1-alpha)**2, 1/2 - alpha

def bossak(M: csc_array, B: csc_array, K: csc_array,
           f: np.ndarray, phi: np.ndarray,
           h: float, beta : float, gamma : float, 
           alpha : float = -0.05,
           a: Union[None,np.ndarray] = None,
           v: Union[None,np.ndarray] = None) -> Tuple[csc_array,np.ndarray]:
    """
    Return left hand side (matrix) and right hand side for one timestep 
    evolution of the Bossak methd. We assume the generic PDE of second 
    order in time 
    
    d**2 phi / dt**2 + b * d phi / dt + D phi = f
    
    where phi is the evolution variable (what we want to solve for), b some 
    parameter D a differential operator purely in space (not time) and f a 
    generic function of space and time. After discretization in space, 
    we convert phi to its discretized equivalent Phi and the equation becomes
    
    M d**2 Phi / dt**2  + B d Phi / dt + K Phi = f
    
    with the discretization of the identity operator M (often called mass 
    matrix), the damping matrix B and matrix of the discretization of the 
    operator D which we call K.
    
    Parameters
    ----------
    M : csc_array
        mass matrx.
    B : csc_array
        damping matrix.
    K : csc_array
        discretization of operator D (e. g. stiffness matrix).
    f : np.ndarray
        generic right hand side function.
    phi : np.ndarray
        variable to solve for.
    beta : float 
        mixing parameter that determines how much of the new acceleration 
        enters the displacement update. Typically calculated with function
        bossak_init and should only be set manually by people witha good 
        understanding of the Bossak-method and time integrators.
    gamma : float
        mixing parameter for velocity update that determines how much of the 
        new acceleration enters the velocity update. Typically calculated with 
        function bossak_init and should only be set manually by people witha 
        good understanding of the Bossak-method and time integrators.
    alpha : float 
        Bossak parameter typically between -0.3 and 0. 0 recovers the Newmark 
        beta method.
    h : int
        time step width.
    a : np.ndarray
        approximate second order derivative in time of phi (acceleration).
    v : np.ndarray
        approximate first order derivative in time of phi (velocity).

    Returns
    -------
    lhs : csc_array
        First one is multiplied with the forces and the stiffness matrix, the 
        others on the right hand side with the mass matrix and the history of 
        the function.
    rhs : np.ndarray
        vector of the right hand side
    """
    #
    lhs = (1-alpha) / (beta*h**2) * M + gamma / (beta*h) * B + K
    #
    rhs = f 
    rhs += M@( (1-alpha) / (beta*h**2) * phi + gamma / (beta*h) * v +\
               (1-alpha) / (2*beta) * a )
    rhs += B@( gamma / (beta*h) * phi + (gamma/beta - 1) * v + (1- gamma/(2*beta))*h*a )
    return lhs, rhs

def bossak_update_derivatives(phi: np.ndarray, phi_old: np.ndarray, 
                              v: np.ndarray, a: np.ndarray,
                              alpha: float, beta: float, gamma: float, 
                              h: float) -> Tuple[np.ndarray,np.ndarray]:
    """
    Update first and second order derivative (in time) of phi after the phi 
    of the new timestep has been found:
    
    a_new = (beta*h)**(-1) *( * (phi - phi_old)/h - v + (1 - 1/(2*beta)*a) ) 
    v_new = v + h * ( (1-gamma)*a + gamma * a_new )
    
    Parameters
    ----------
    phi : np.ndarray
        variable to evolve in time with the Bossak scheme.
    phi_old : np.ndarray
        phi of previous timestep.
    v : np.ndarray
        approximate first order derivative in time of phi (velocity).
    a : np.ndarray
        approximate second order derivative in time of phi (acceleration).
    beta : float 
        mixing parameter that determines how much of the new acceleration 
        enters the displacement update. Typically calculated with function
        bossak_init and should only be set manually by people witha good 
        understanding of the Bossak-method and time integrators.
    gamma : float
        mixing parameter for velocity update that determines how much of the 
        new acceleration enters the velocity update. Typically calculated with 
        function bossak_init and should only be set manually by people witha 
        good understanding of the Bossak-method and time integrators.
    alpha : float 
        Bossak parameter typically between -0.3 and 0. 0 recovers the Newmark 
        beta method.
    h : int
        time step width.

    Returns
    -------
    lhs : csc_array
        First one is multiplied with the forces and the stiffness matrix, the 
        others on the right hand side with the mass matrix and the history of 
        the function.
    rhs : np.ndarray
        vector of the right hand side
    """
    a_new = 1 / (beta*h**2) * (phi - phi_old) - 1 / (beta*h) * v + (1 - 1/(2*beta)*a) 
    v_new = v + h * ( (1-gamma)*a + gamma * a_new )
    return a_new, v_new