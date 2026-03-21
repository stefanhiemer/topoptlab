# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, Union

import numpy as np

from topoptlab.optimizer.stepsize import barzilai_borwein_short

def gradient_descent(x: np.ndarray, 
                     dobj: np.ndarray, 
                     xmin: Union[float,np.ndarray], 
                     xmax: Union[float,np.ndarray], 
                     xold: Union[None,np.ndarray] = None, 
                     dobjold: Union[None,np.ndarray] = None,
                     stepsize_func: Callable = barzilai_borwein_short,
                     stepsize_kw: Dict = {},
                     move: int = 0.1,
                     **kwargs: Any) -> np.ndarray:
    """
    Bounded gradient descent with move limit. 
    
    Parameters
    ----------
    x : np.ndarray, shape (n)
        design variablesof the current iteration.
    dobj : np.array, shape (n)
        gradient of objective function with respect to design variables.
    xmin : np.ndarray, shape (n)
        minimum value of design variables.
    xmax : np.ndarray, shape (n)
        maximum value of design variables.
    xold : None or np.ndarray, shape (n)
        design variables  of the previous iteration.
    dobjold : None or np.array, shape (n)
        gradient of objective function with respect to design variables of 
        the previous iteration.
    stepsize_func : str
        function to determine step size. 
    stepsize_kw : dict
        dictionary containing arguments for the stepsize_func. x, dobj and its 
        older versions are automatically provided.
    move: float
        maximum change allowed in each design variable.
        
    Returns
    -------
    xnew : np.array, shape (n)
        updated optimization variables.
    """
    #
    alpha = stepsize_func(x=x,
                          xold=xold,
                          fgrad=dobj,
                          fgradold=dobjold,
                          **stepsize_kw)
    #
    if np.isclose(alpha, 0.) or np.isinf(alpha) or np.isnan(alpha):
        raise ValueError("No step size could be found: ",alpha)
    #
    xnew = x-alpha*dobj
    # apply bounding and move limit
    xnew[:] = np.maximum(xmin, 
                         np.maximum(x-move, 
                                    np.minimum(xmax, 
                                               np.minimum(x+move, 
                                                          xnew))))
    return xnew
