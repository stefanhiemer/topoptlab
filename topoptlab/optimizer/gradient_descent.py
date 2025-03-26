from __future__ import division

import numpy as np

projections = [2,3,4,5]
filters = [0,1]

def gradient_descent(x, dc, xmin, xmax, 
                     el_flags, move=0.2):
    """
    Simple gradient respect that respects lower and upper bounds for the design 
    variables.
    
    Parameters
    ----------
    x : np.ndarray, shape (nel)
        element densities for topology optimization of the current iteration.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    xmin : np.ndarray, shape (nel)
        element densities for topology optimization of the current iteration.
    xmax : np.ndarray, shape (nel)
        element densities for topology optimization of the current iteration.
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements. 
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
    move: float
        maximum change allowed in the density of a single element.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    #
    dx = move/dc.max()
    #
    x[:] = np.maximum(xmin, 
                      np.maximum(x-move, 
                                 np.minimum(xmax, 
                                            np.minimum(x+move, x+dx*dc))))
    # passive element update
    if el_flags is not None:
        x[el_flags==1] = 0
        x[el_flags==2] = 1
    return x
