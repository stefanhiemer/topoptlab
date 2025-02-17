from topoptlab.elements.poisson_2d import lk_poisson_2d

import numpy as np

def mass_3d():
    """
    Create mass matrix for 3D with trilinear hexahedral Lagrangian 
    elements. 
        
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    return np.array([[1/9, 1/18, 1/36, 1/18],
                     [1/18, 1/9, 1/18, 1/36],
                     [1/36, 1/18, 1/9, 1/18],
                     [1/18, 1/36, 1/18, 1/9]])
