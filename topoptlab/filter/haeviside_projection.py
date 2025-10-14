# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple

import numpy as np
from scipy.optimize import root_scalar

def find_eta(eta0: float, xTilde: np.ndarray, beta: float, volfrac: float,
             root_args: Dict = {"fprime": True,
                                "method": "newton",
                                "maxiter": 1000,
                                "bracket": [-1/2,1/2]},
             **kwargs: Any) -> float:
    """
    Find volume preserving eta for the element-wiser elaxed Haeviside 
    as has been done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    # unfortunately scipy.optimize needs f to change sign between the
    # respective ends of the brackets, therefor the eta found by this function
    # is offset by -1/2 to the value later used
    result = root_scalar(f=_root_func, x0=eta0-1/2, args=(xTilde,beta,volfrac),
                         x1=0.,
                         fprime=True, method="newton", maxiter=1000, bracket=[-1/2,1/2])
    #
    if result.converged:
        return result.root+1/2
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _root_func(eta: float, xTilde: np.ndarray, 
               beta: float, volfrac: float) -> Tuple[float,float]:
    """
    Function whose root is the volume preserving threshold.

    Parameters
    ----------
    eta : float
        current threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.

    Returns
    -------
    res : float
        value of current volume fraction - intended volume fraction.
    gradient : float
        gradient for Newton procedure

    """
    #
    eta = eta + 1/2
    #
    xPhys = eta_projection(eta=eta,xTilde=xTilde,beta=beta)
    # terms
    tanh_bn = np.tanh(beta * eta)
    tanh_b1n = np.tanh(beta * (1 - eta))
    tanh_bx_n = np.tanh(beta * (xTilde - eta))
    tanh_bn_x = np.tanh(beta * (eta - xTilde))

    sech2_bn = 1 - tanh_bn**2
    sech2_bx_n = 1 - tanh_bx_n**2
    sech2_b1n = 1 - tanh_b1n**2
    #
    denom1 = tanh_bn + tanh_b1n
    denom2 = denom1 ** 2
    #
    term1 = -sech2_bx_n
    term2 =  sech2_bn * (tanh_b1n + tanh_bn_x) + sech2_b1n * (tanh_bn + tanh_bx_n)
    return xPhys.mean()-volfrac, beta*(term1/denom1 + term2/denom2).mean()

def eta_projection(eta: float, xTilde: np.ndarray, 
                   beta: float) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    return (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))