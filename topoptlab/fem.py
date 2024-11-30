import numpy as np

def update_indices(indices,fixed,mask):
    """
    Update the indices for the stiffness matrix construction by kicking out
    the fixed degrees of freedom and renumbering the indices.

    Parameters
    ----------
    indices : np.array
        indices of degrees of freedom used to construct the stiffness matrix.
    fixed : np.array
        indices of fixed degrees of freedom.
    mask : np.array
        mask to kick out fixed degrees of freedom.

    Returns
    -------
    indices : np.arrays
        updated indices.

    """
    val, ind = np.unique(indices,return_inverse=True)
    
    _mask = ~np.isin(val, fixed)
    val[_mask] = np.arange(_mask.sum())
    
    return val[ind][mask]

def lk_linear_elast_2D(E=1,nu=0.3):
    """
    Create element stiffness matrix for 2D linear elasticity with bilinear
    quadratic elements.
    
    
    Returns
    -------
    Ke : np.array, shape (8,8)
        element stiffness matrix.
        
    """
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    Ke = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return (Ke)

def lk_Poisson_2D():
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadratic elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.array, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = np.array([[2/3, -1/6, -1/3, -1/6,],
                   [-1/6, 2/3, -1/6, -1/3],
                   [-1/3, -1/6, 2/3, -1/6],
                   [-1/6, -1/3, -1/6, 2/3]])
    return (Ke)

def lk_screened_Poisson_2D(Rmin):
    Ke = (Rmin**2) * np.array([[4, -1, -2, -1],
                                [-1, 4, -1, -2],
                                [-2, -1, 4, -1],
                                [-1, -2, -1, 4]])/6 + \
                     np.array([[4, 2, 1, 2],
                               [2, 4, 2, 1],
                               [1, 2, 4, 2],
                               [2, 1, 2, 4]])/36
    return (Ke)
