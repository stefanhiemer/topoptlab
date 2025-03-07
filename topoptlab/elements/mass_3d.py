import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.trilinear_hexahedron import jacobian,shape_functions

def _lm_mass_3d(xe,p=1.0,
                quadr_method="gauss-legendre",
                nquad=2):
    """
    Create element mass matrix in 3D with trilinear hexahedral elements. 
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    p : np.ndarray of shape (nels) or (1)
        density of element
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of 
        quadrature points and weights. Check function get_integrpoints for 
        available options.
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,8,8)
        element stiffness matrix.
        
    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    #
    if isinstance(p,float) or (p.shape[0] == 1 and xe.shape[0] !=1):
        p = np.full(xe.shape[0], p)
    #
    x,w=get_integrpoints(ndim=3,nq=nquad,method=quadr_method)
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, 3,axis=1)]
    #
    N = shape_functions(xi=xi,eta=eta,zeta=zeta)
    #
    integral = N[None,:,:,None]@N[None,:,:,None,].transpose([0,1,3,2])
    # multiply by determinant
    #integral = integral * detJ[:,None,None]
    #
    Ke = (w[:,None,None]*integral).sum(axis=1)
    # 
    #J = jacobian(xi,eta,xe,all_elems=False)
    #det = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    return p[:,None,None] * Ke

def lm_mass_3d(p=1.0):
    """
    Create element mass matrix in 3D with trilinear hexahedral elements. 
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    return p*np.array([[8/27,4/27,2/27,4/27,4/27,2/27,1/27,2/27],
                       [4/27,8/27,4/27,2/27,2/27,4/27,2/27,1/27],
                       [2/27,4/27,8/27,4/27,1/27,2/27,4/27,2/27],
                       [4/27,2/27,4/27,8/27,2/27,1/27,2/27,4/27],
                       [4/27,2/27,1/27,2/27,8/27,4/27,2/27,4/27],
                       [2/27,4/27,2/27,1/27,4/27,8/27,4/27,2/27],
                       [1/27,2/27,4/27,2/27,2/27,4/27,8/27,4/27],
                       [2/27,1/27,2/27,4/27,4/27,2/27,4/27,8/27]])
