import numpy as np

def barzilai_borwein(x, dobj, 
                     xold, dobjold,
                     xmin, xmax, 
                     el_flags, 
                     mode="long",
                     move=0.1):
    """
    Barzilai-Borwain gradient descent that respects lower and upper bounds for 
    the design variables.
    
    Parameters
    ----------
    x : np.ndarray, shape (nel)
        element densities for topology optimization of the current iteration.
    dobj : np.array, shape (nel)
        gradient of objective function with respect to element 
        densities.
    xold : np.ndarray, shape (nel)
        element densities for topology optimization of the previous iteration.
    dobjold : np.array, shape (nel)
        gradient of objective function with respect to element densities of 
        the previous iteration.
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
    xnew = np.zeros(x.shape)
    # calculate step size
    dx = x-xold
    dg = dobj-dobjold
    #
    if mode == "long":
        alpha = dx.dot(dx) / dx.dot(dg)
    elif mode == "short":
        alpha = dx.dot(dg) / dg.dot(dg)
    elif mode == "stabilized":
        alpha = np.minimum(dx.dot(dx) / dx.dot(dg), 
                           np.sqrt( dx.dot(dx) ))
        print(alpha)
    if np.isclose(alpha, 0.) or np.isinf(alpha):
        raise ValueError("No step size could be found.")
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
