# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Union

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
                                fgradold: Union[None,np.ndarray],
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
    if fgradold is None:
        return 1e-6
    else:
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
    if fgradold is None:
        return 1e-5
    else:
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
    if fgradold is None:
        return 1e-5
    else:
        # calculate step size
        dx = x-xold
        dg = fgrad-fgradold
        print(dx)
        print(dg)
        return dx.dot(dg) / dg.dot(dg)

def check_wolfe_conditions(f_prev: float,
                           f_new: float,
                           g_prev: np.ndarray,
                           p: np.ndarray,
                           alpha: float,
                           g_new: np.ndarray | None = None,
                           c1: float = 1e-4,
                           c2: float = 0.9,
                           strong: bool = False, 
                           **kwargs: Any
                           ) -> bool:
    """
    Check Armijo, Wolfe, or strong Wolfe conditions. Different checks are done
    depending on the arguments:
        
        - If g_new is None:
            only the Armijo (sufficient decrease) condition is checked.
        - If g_new is provided and strong=False:
            standard Wolfe conditions are checked.
        - If g_new is provided and strong=True:
            strong Wolfe conditions are checked.

    1) Armijo / sufficient decrease:
    
        f_new <= f_prev + c1 * alpha * (g_prev^T p)

    2) Standard Wolfe curvature condition:
    
        g_new^T p >= c2 * (g_prev^T p)

    3) Strong Wolfe curvature condition:
    
        |g_new^T p| <= c2 * |g_prev^T p|

    Thus:
    - Wolfe conditions = Armijo + standard curvature
    - Strong Wolfe conditions = Armijo + strong curvature

    Parameters
    ----------
    f_prev : float
        objective value at previous iterate, f(x_k).
    f_new : float
        objective value at trial/new iterate, f(x_k + alpha * p).
    g_prev : np.ndarray
        gradient at previous iterate, grad f(x_k).
    p : np.ndarray
        search direction of shape (n).
    alpha : float
        step length.
    g_new : np.ndarray | None, optional
        Gradient at trial/new iterate, grad f(x_k + alpha * p). If None, only 
        Armijo is checked.
    c1 : float, optional
        Armijo parameter.
    c2 : float, optional
        curvature parameter.
    strong : bool, optional
        if True and g_new is provided, check strong Wolfe instead of standard Wolfe.

    Returns
    -------
    line_flag: bool
        True if the requested condition(s) hold, False otherwise.
    """
    # check line search parameters
    if not (0. < c1 < 1.):
        raise ValueError("Require 0 < c1 < 1.")
    if g_new is not None and not (c1 < c2 < 1.0):
        raise ValueError("Require 0 < c1 < c2 < 1 when checking Wolfe conditions.")
    #
    g_prev_dot_p = np.dot(g_prev, p)
    # check p is a descent direction
    if g_prev_dot_p >= 0.:
        armijo = False
    else:
        # check Armijo / sufficient decrease
        armijo = f_new <= f_prev + c1 * alpha * g_prev_dot_p
    # if armijo is False, no need to check anything
    if armijo and not (g_new is None):
        #
        if strong:
            curvature = np.abs(np.dot(g_new, p)) <= c2 * np.abs(g_prev_dot_p)
        else:
            curvature = np.dot(g_new, p) >= c2 * g_prev_dot_p
    else:
        curvature=True
    #
    return armijo and curvature
