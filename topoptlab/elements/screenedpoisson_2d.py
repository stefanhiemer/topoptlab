from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.mass_2d import mass_2d 

def lk_screened_poisson_2d(k):
    """
    Create matrix for 2D screened Poisson equation with bilinear quadrilateral 
    elements. Taken from the 88 lines code and slightly modified.
    
    Parameters
    ----------
    k : float
        analogous to heat conductivity. k is the squared filter radius if used for the 
        'Helmholtz' filter.
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = k * lk_poisson_2d() + mass_2d()
    return Ke

def lk_screened_poisson_aniso_2d(k):
    """
    Create matrix for 2D screened Poisson equation with bilinear quadrilateral 
    elements. Taken from the 88 lines code and slightly modified.
    
    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        analogous to anisotropic heat conductivity. If isotropic k would be 
        [[k,0],[0,k]] and k is the squared filter radius if used for the 
        'Helmholtz' filter.
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = lk_poisson_2d(k) + mass_2d()
    return Ke
