from topoptlab.elements.poisson_2d import lk_poisson_2d

import numpy as np

def mass_2d(rmin):
    """
    Create mass matrix for 2D with bilinear quadrilateral Lagrangian 
    elements. Taken from the 88 lines code and slightly modified.
    
    Parameters
    ----------
    rmin : float
        filter radius.
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[1/9, 1/18, 1/36, 1/18],
                     [1/18, 1/9, 1/18, 1/36],
                     [1/36, 1/18, 1/9, 1/18],
                     [1/18, 1/36, 1/18, 1/9]])
