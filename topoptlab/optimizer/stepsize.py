# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def constant(step_size: float,
             **kwargs: Any) -> float:
    """
    Returns a fixed step size.

    This function is provided for interface compatibility with functions for 
    adaptive step-size rules. It returns the input value unchanged.


    Parameters
    ----------
    step_size : float
        prescribed step size to use.

    Returns
    -------
    step_size : float
        constant step size.

    """
    
    return step_size

def barzilai_borwein_stabilized(x: np.ndarray, 
                                fgrad: np.ndarray, 
                                xold: np.ndarray, 
                                fgradold: np.ndarray,
                                **kwargs: Any
                                ) -> float:
    """
    Computes the stabilized Barzilai-Borwein step size:
        
    Burdakov, Oleg, Yuhong Dai, and Na Huang. "Stabilized barzilai-borwein 
    method." Journal of Computational Mathematics (2019): 916-936.
    
    We recommend only using it if the original formulas have failed.
    
    Parameters
    ----------
    x : np.ndarray, shape (nel)
        design variables of the current iteration.
    fgrad : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables.
    xold : np.ndarray, shape (nel)
        design variables  of the previous iteration.
    fgradold : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables of 
        the previous iteration.

    Returns
    -------
    step_size : float
        stabilized Barzilai-Borwein step size.

    """
    # calculate step size
    dx = x-xold
    dg = fgrad-fgradold
    return np.minimum(dx.dot(dx) / dx.dot(dg), 
                       np.sqrt( dx.dot(dx) ))

def barzilai_borwein_long(x: np.ndarray, 
                          fgrad: np.ndarray, 
                          xold: np.ndarray, 
                          fgradold: np.ndarray,
                          **kwargs: Any
                          ) -> float:
    """
    Compute the long Barzilai-Borwein step size.
    
    Barzilai, Jonathan, and Jonathan M. Borwein. "Two-point step size gradient 
    methods." IMA journal of numerical analysis 8.1 (1988): 141-148.
    
    For explanation, have a look at wikipedia. Be warned that this function 
    assumes that ``x`` and ``fgrad`` change at each iteration.  

    Parameters
    ----------
    x : np.ndarray, shape (nel)
        design variables of the current iteration.
    fgrad : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables.
    xold : np.ndarray, shape (nel)
        design variables  of the previous iteration.
    fgradold : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables of 
        the previous iteration.

    Returns
    -------
    step_size : float
        long Barzilai-Borwein step size.

    """
    # calculate step size
    dx = x-xold
    dg = fgrad-fgradold
    return dx.dot(dx) / dx.dot(dg)

def barzilai_borwein_short(x: np.ndarray, 
                           fgrad: np.ndarray, 
                           xold: np.ndarray, 
                           fgradold: np.ndarray,
                           **kwargs: Any
                           ) -> float:
    """
    Compute the short Barzilai-Borwein step size.
    
    Barzilai, Jonathan, and Jonathan M. Borwein. "Two-point step size gradient 
    methods." IMA journal of numerical analysis 8.1 (1988): 141-148.
    
    For explanation, have a look at wikipedia. Be warned that this function 
    assumes that``fgrad`` changes at each iteration.  

    Parameters
    ----------
    x : np.ndarray, shape (nel)
        design variables of the current iteration.
    fgrad : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables.
    xold : np.ndarray, shape (nel)
        design variables of the previous iteration.
    fgradold : np.ndarray, shape (nel)
        gradient of objective function with respect to design variables of 
        the previous iteration.

    Returns
    -------
    step_size : float
        short Barzilai-Borwein step size.

    """
    # calculate step size
    dx = x-xold
    dg = fgrad-fgradold
    return dx.dot(dg) / dg.dot(dg)
