# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union,Any

import numpy as np

def gradient_descent(x: np.ndarray, 
                     dobj: np.ndarray, 
                     xmin: Union[float,np.ndarray], 
                     xmax: Union[float,np.ndarray], 
                     stepsize: float = 1e-4,
                     move: int = 0.1,
                     **kwargs: Any) -> np.ndarray:
    """
    Standard gradient descent. Just for demonstration, teaching and testing 
    purposes. Do not use in series research endeavours.
    
    Parameters
    ----------
    x : np.ndarray, shape (nel)
        design variables of the current iteration.
    dobj : np.array, shape (nel)
        gradient of objective function with respect to design variables.
    xmin : float or np.ndarray, shape (nel)
        minimum value of design variables.
    xmax : float or np.ndarray, shape (nel)
        maximum value of design variables.
    stepsize : float
        step size 
    move: float
        maximum change allowed in each design variable.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updated design variables.
    """
    #
    xnew = np.zeros(x.shape)
    #
    xnew[:] = np.maximum(xmin, 
                         np.maximum(x-move, 
                                    np.minimum(xmax, 
                                               np.minimum(x+move, 
                                                          x-stepsize*dobj))))
    return xnew

def barzilai_borwein(x: np.ndarray, dobj: np.ndarray, 
                     xold: np.ndarray, dobjold: np.ndarray,
                     xmin: Union[float,np.ndarray], 
                     xmax: Union[float,np.ndarray], 
                     el_flags: Union[None,np.ndarray], 
                     step_mode: str = "long",
                     move: int = 0.1,
                     **kwargs: Any) -> np.ndarray:
    """
    Barzilai-Borwain gradient descent that respects lower and upper bounds for 
    the design variables. It offers the standard step size methods from the 
    original paper 
    
    Barzilai, Jonathan, and Jonathan M. Borwein. "Two-point step size gradient 
    methods." IMA journal of numerical analysis 8.1 (1988): 141-148.
    
    and also offers a stabilized stepping method taken from
    
    Burdakov, Oleg, Yuhong Dai, and Na Huang. "Stabilized barzilai-borwein 
    method." Journal of Computational Mathematics (2019): 916-936.
    
    Parameters
    ----------
    x : np.ndarray, shape (nel)
        design variablesof the current iteration.
    dobj : np.array, shape (nel)
        gradient of objective function with respect to design variables.
    xold : np.ndarray, shape (nel)
        design variables  of the previous iteration.
    dobjold : np.array, shape (nel)
        gradient of objective function with respect to design variables of 
        the previous iteration.
    xmin : np.ndarray, shape (nel)
        minimum value of design variables.
    xmax : np.ndarray, shape (nel)
        maximum value of design variables.
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements. 
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
    step_method : str
        method to determine step size. Either "long","short" or "stabilized".
    move: float
        maximum change allowed in each design variable.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.

    """
    #
    xnew = np.zeros(x.shape)
    # calculate step size
    dx = x-xold
    dg = dobj-dobjold
    #
    if step_mode == "long":
        alpha = dx.dot(dx) / dx.dot(dg)
    elif step_mode == "short":
        alpha = dx.dot(dg) / dg.dot(dg)
    elif step_mode == "stabilized":
        alpha = np.minimum(dx.dot(dx) / dx.dot(dg), 
                           np.sqrt( dx.dot(dx) ))
    if np.isclose(alpha, 0.) or np.isinf(alpha):
        raise ValueError("No step size could be found: ",alpha)
    #
    xnew[:] = np.maximum(xmin, 
                         np.maximum(x-move, 
                                    np.minimum(xmax, 
                                               np.minimum(x+move, x-alpha*dobj))))
    # passive/active element update
    if el_flags is not None:
        xnew[el_flags==1] = 0
        xnew[el_flags==2] = 1
    return xnew
