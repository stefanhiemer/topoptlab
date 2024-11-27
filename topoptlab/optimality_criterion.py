from __future__ import division

import numpy as np

from topoptlab.filters import AMfilter
    
def oc_top88(nelx, nely, x, volfrac, dc, dv, g, pass_el,
             move = 0.2,l1=0.,l2=1e9):
    """
    Optimality criteria method (section 2.2 in paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Optimality criteria method (section 2.2 in paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Overtaken and adapted from the 
    
    165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    pass_el : None or np.array 
        array who contains indices used for un/masking passive elements. 0 
        means an active element that is part of the optimization, 1 and 2 
        indicate empty and full elements which are not part of the 
        optimization.
    move: float
        maximum change allowed in the density of a single element.
    l1: float
        starting guess for the lower part of the bisection algorithm.
    l2: float
        starting guess for the upper part of the bisection algorithm.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    
    # reshape to perform vector operations
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
        
        # passive element update
        if pass_el is not None:
            xnew[pass_el==1] = 0
            xnew[pass_el==2] = 1
        gt=g+np.sum((dv*(xnew-x))) #g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        
    return (xnew, gt)
