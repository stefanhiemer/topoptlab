import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import jacobian,shape_functions_dxi

def _lk_poisson_2d(xe,k,
                   quadr_method="gauss-legendre",
                   t=np.array([1.]),
                   nquad=2):
    """
    Create element stiffness matrix for 2D Laplacian operator with bilinear
    quadrilateral elements. 
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    k : np.ndarray, shape (nels,3,3) or 
        conductivity tensor or something equivalent.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of 
        quadrature points and weights. Check function get_integrpoints for 
        available options. 
    t : np.ndarray of shape (nels) or (1)
        thickness of element
    nquad : int
        number of quadrature points
    Returns
    -------
    Ke : np.ndarray, shape (nels,4,4)
        element stiffness matrix.
        
    """
    #
    if len(k.shape) == 2:
        k = k[None,:,:]
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    gradN = shape_functions_dxi(xi=xi,eta=eta)
    #
    integral = gradN[None,:,:,:]@k[:,None,:,:]@gradN[None,:,:,:].transpose([0,1,3,2])
    # multiply by determinant
    #integral = integral * detJ[:,None,None]
    #
    Ke = (w[:,None,None]*integral).sum(axis=1)
    Ke = t[:,None,None] * Ke  
    # 
    #J = jacobian(xi,eta,xe,all_elems=False)
    #det = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    return Ke

def lk_poisson_2d(k=1):
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadrilateral elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return k*np.array([[2/3, -1/6, -1/3, -1/6],
                       [-1/6, 2/3, -1/6, -1/3],
                       [-1/3, -1/6, 2/3, -1/6],
                       [-1/6, -1/3, -1/6, 2/3]])

def lk_poisson_aniso_2d(k):
    """
    Create element stiffness matrix for anisotropic 2D Poisson with bilinear
    quadrilateral elements. 
    
    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        anisotropic heat conductivity. If isotropic k would be [[k,0],[0,k]]
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = np.array([[k[0,0]/3 + k[0,1]/4 + k[1,0]/4 + k[1,1]/3, 
                    -k[0,0]/3 + k[0,1]/4 - k[1,0]/4 + k[1,1]/6, 
                    -k[0,0]/6 - k[0,1]/4 - k[1,0]/4 - k[1,1]/6, 
                    k[0,0]/6 - k[0,1]/4 + k[1,0]/4 - k[1,1]/3], 
                   [-k[0,0]/3 - k[0,1]/4 + k[1,0]/4 + k[1,1]/6, 
                    k[0,0]/3 - k[0,1]/4 - k[1,0]/4 + k[1,1]/3, 
                    k[0,0]/6 + k[0,1]/4 - k[1,0]/4 - k[1,1]/3, 
                    -k[0,0]/6 + k[0,1]/4 + k[1,0]/4 - k[1,1]/6], 
                   [-k[0,0]/6 - k[0,1]/4 - k[1,0]/4 - k[1,1]/6, 
                    k[0,0]/6 - k[0,1]/4 + k[1,0]/4 - k[1,1]/3, 
                    k[0,0]/3 + k[0,1]/4 + k[1,0]/4 + k[1,1]/3, 
                    -k[0,0]/3 + k[0,1]/4 - k[1,0]/4 + k[1,1]/6], 
                   [k[0,0]/6 + k[0,1]/4 - k[1,0]/4 - k[1,1]/3, 
                    -k[0,0]/6 + k[0,1]/4 + k[1,0]/4 - k[1,1]/6, 
                    -k[0,0]/3 - k[0,1]/4 + k[1,0]/4 + k[1,1]/6, 
                    k[0,0]/3 - k[0,1]/4 - k[1,0]/4 + k[1,1]/3]])
    return Ke