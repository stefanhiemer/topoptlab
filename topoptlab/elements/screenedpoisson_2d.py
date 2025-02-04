from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.mass_2d import mass_2d 

def lk_screened_poisson_2d(rmin):
    """
    Create matrix for 2D screened Poisson equation with bilinear quadrilateral 
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
    Ke = (rmin**2) * lk_poisson_2d() + mass_2d()
    return Ke
