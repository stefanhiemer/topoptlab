# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.interpolate import CubicHermiteSpline

def gray_indicator(x: np.ndarray) -> np.ndarray:
    """
    Gray level indicator to measure discreteness of the designs as in Eq. 41 
    in 
    
    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.
    
    For a design with all densities at 1/2 returns 1, if all densities either
    1 or 0 it returns 0.

    Parameters
    ----------
    x : np.ndarray shape (n) or shape (n,nmats)
        design variables. 

    Returns
    -------
    indicator : np.ndarray of shape () or shape (nmats)
        intermediate density indicator.

    """
    return 4*(x*(1-x)).mean()

def level_indicator(x: np.ndarray,
                    x_i: [None,np.ndarray] = None,
                    nlevels: None | int = None) -> np.ndarray:
    """
    Level indicator to measure discreteness of multi-level designs. Generalizes
    gray_indicator to the case where x is expected to take more than two
    discrete values x_0, x_1, ..., x_N. The indicator is constructed as a
    piecewise cubic Hermite spline with peaks of 1 halfway between adjacent
    levels and zeros at the levels themselves. Returns 0 when all densities
    coincide with a discrete level and approaches 1 when all densities sit
    exactly halfway between two adjacent levels.

    Supply either x_i or nlevels, not both.

    Parameters
    ----------
    x : np.ndarray of shape (n,) or (n, nmats)
        design variables.
    x_i : np.ndarray of shape (nlevels,) or None
        explicit discrete target levels, e.g. np.array([0., 0.5, 1.]).
    nlevels : int or None
        number of equally spaced levels in [0, 1].

    Returns
    -------
    indicator : np.ndarray of shape () or (nmats,)
        intermediate density indicator in [0, 1].

    """
    if x_i is None and isinstance(nlevels,int):
        herm = CubicHermiteSpline(x=np.linspace(0,1/nlevels,3),
                                  y=np.array([0,1,0]),
                                  dydx=np.zeros(3),
                                  extrapolate="periodic")
    elif isinstance(x_i,np.ndarray) and len(x_i.shape)==1:
        # midpoints
        x_mid = (x_i[1:]+x_i[:-1])/2
        x_knots = np.column_stack((x_i, np.append(x_mid,np.zeros(1)))).flatten()[:-1]
        #
        herm = CubicHermiteSpline(x=x_knots,
                                  y=np.tile([0,1],x_i.shape)[:-1],
                                  dydx=np.zeros(2*x_i.shape[0]-1),
                                  extrapolate=None)
    else:
        raise NotImplementedError("Input inconsistent.")
    return np.array([herm(x[:,i]).mean() for i in np.arange(x.shape[1])])
