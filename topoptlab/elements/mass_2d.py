import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import shape_functions

def _lm_mass_2d(xe,
                p=np.array([1.]),
                t=np.array([1.]),
                quadr_method="gauss-legendre",
                nquad=2):
    """
    Create element mass matrix in 2D with bilinear quadrilateral elements. 
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    p : np.ndarray of shape (nels) or (1)
        density of element
    t : np.ndarray of shape (nels) or (1)
        thickness of element
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of 
        quadrature points and weights. Check function get_integrpoints for 
        available options. 
    nquad : int
        number of quadrature points
    Returns
    -------
    Ke : np.ndarray, shape (nels,4,4)
        element stiffness matrix.
        
    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    if isinstance(p,float) or (p.shape[0] == 1 and xe.shape[0] !=1):
        p = np.full(xe.shape[0], p)
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    N = shape_functions(xi=xi,eta=eta)
    #
    integral = N[None,:,:,None]@N[None,:,:,None,].transpose([0,1,3,2])
    # multiply by determinant
    #integral = integral * detJ[:,None,None]
    #
    Ke = (w[:,None,None]*integral).sum(axis=1)
    # 
    #J = jacobian(xi,eta,xe,all_elems=False)
    #det = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    return t[:,None,None] * p[:,None,None] * Ke

def lm_mass_symfem(p=1.,t=1.):
    """
    Create mass matrix for 2D with bilinear quadrilateral Lagrangian 
    elements. 
    
    Parameters
    ----------
    p : float
        density of element
    t : float
        thickness of element
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    
    return p*t*np.array([[4/9,2/9,1/9,2/9],
                         [2/9,4/9,2/9,1/9],
                         [1/9,2/9,4/9,2/9],
                         [2/9,1/9,2/9,4/9]])

def lm_mass_2d():
    """
    Create mass matrix for 2D with bilinear quadrilateral Lagrangian 
    elements. Taken from the 88 lines code and slightly modified.
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[1/9, 1/18, 1/36, 1/18],
                     [1/18, 1/9, 1/18, 1/36],
                     [1/36, 1/18, 1/9, 1/18],
                     [1/18, 1/36, 1/18, 1/9]])
