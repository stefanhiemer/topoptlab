from topoptlab.elements.poisson_3d import lk_poisson_3d,lk_poisson_aniso_3d
from topoptlab.elements.mass_3d import lm_mass_3d 

def lk_screened_poisson_3d(k):
    """
    Create matrix for 3D screened Poisson equation with trilinear hexahedral 
    elements.
    
    Parameters
    ----------
    k : float
        analogous to heat conductivity. k is the squared filter radius if used 
        for the 'Helmholtz' filter.
        
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    Ke = lk_poisson_3d(k) + lm_mass_3d()
    return Ke

def lk_screened_poisson_aniso_3d(k):
    """
    Create matrix for 3D screened Poisson equation with trilinear hexahedral 
    elements. 
    
    Parameters
    ----------
    k : np.ndarray, shape (3,3)
        analogous to anisotropic heat conductivity. If isotropic k would be 
        [[k,0,0],[0,k,0],[0,0,k]] and k is the squared filter radius if used for the 
        'Helmholtz' filter.
        
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    Ke = lk_poisson_aniso_3d(k) + lm_mass_3d()
    return Ke
