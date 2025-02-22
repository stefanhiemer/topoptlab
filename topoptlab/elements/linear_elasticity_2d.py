import numpy as np 

from topoptlab.elements.bilinear_quadrilateral import bmatrix
from topoptlab.fem import get_integrpoints

def _lk_linear_elast_2d(xe,c,
                        quadr_method="gauss-legendre",
                        t=np.array([1.]),
                        nquad=2):
    """
    Create element stiffness matrix for 2D linear elasticity with 
    bilinear quadrilateral Lagrangian elements in plane stress.
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    c : np.ndarray, shape (nels,3,3) or 
        stiffness tensor.
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
    Ke : np.ndarray, shape (nels,8,8)
        element stiffness matrix.
        
    """
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    nel = xe.shape[0]
    nq =w.shape[0]
    # 
    B,detJ = bmatrix(xi, eta, xe, all_elems=True, return_detJ=True)
    print(detJ.shape)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]@B
    #
    Ke = (w[:,None,None]*integral).sum(axis=1)
    # multiply jacobi determinant and thickness
    Ke = t[:,None,None] * Ke# * detJ[:,None,None] 
    return Ke

def lk_linear_elast_2d(E=1,nu=0.3):
    """
    Create element stiffness matrix for 2D isotropic linear elasticity with 
    bilinear quadrilateral Lagrangian elements in plane stress. 
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson' ratio.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
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
    return Ke

def lk_linear_elast_aniso_2d(c):
    """
    Create element stiffness matrix for 2D anisotropic linear elasticity with 
    bilinear quadrilateral elements. 
    
    Parameters
    ----------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    Ke = np.array([[c[0,0]/3 + c[0,2]/4 + c[2,0]/4 + c[2,2]/3, 
                    c[0,1]/4 + c[0,2]/3 + c[2,1]/3 + c[2,2]/4, 
                    -c[0,0]/3 + c[0,2]/4 - c[2,0]/4 + c[2,2]/6, 
                    c[0,1]/4 - c[0,2]/3 + c[2,1]/6 - c[2,2]/4, 
                    -c[0,0]/6 - c[0,2]/4 - c[2,0]/4 - c[2,2]/6, 
                    -c[0,1]/4 - c[0,2]/6 - c[2,1]/6 - c[2,2]/4, 
                    c[0,0]/6 - c[0,2]/4 + c[2,0]/4 - c[2,2]/3, 
                    -c[0,1]/4 + c[0,2]/6 - c[2,1]/3 + c[2,2]/4], 
                   [c[1,0]/4 + c[1,2]/3 + c[2,0]/3 + c[2,2]/4, 
                    c[1,1]/3 + c[1,2]/4 + c[2,1]/4 + c[2,2]/3, 
                    -c[1,0]/4 + c[1,2]/6 - c[2,0]/3 + c[2,2]/4, 
                    c[1,1]/6 - c[1,2]/4 + c[2,1]/4 - c[2,2]/3, 
                    -c[1,0]/4 - c[1,2]/6 - c[2,0]/6 - c[2,2]/4, 
                    -c[1,1]/6 - c[1,2]/4 - c[2,1]/4 - c[2,2]/6, 
                    c[1,0]/4 - c[1,2]/3 + c[2,0]/6 - c[2,2]/4, 
                    -c[1,1]/3 + c[1,2]/4 - c[2,1]/4 + c[2,2]/6], 
                   [-c[0,0]/3 - c[0,2]/4 + c[2,0]/4 + c[2,2]/6, 
                    -c[0,1]/4 - c[0,2]/3 + c[2,1]/6 + c[2,2]/4, 
                    c[0,0]/3 - c[0,2]/4 - c[2,0]/4 + c[2,2]/3, 
                    -c[0,1]/4 + c[0,2]/3 + c[2,1]/3 - c[2,2]/4, 
                    c[0,0]/6 + c[0,2]/4 - c[2,0]/4 - c[2,2]/3, 
                    c[0,1]/4 + c[0,2]/6 - c[2,1]/3 - c[2,2]/4, 
                    -c[0,0]/6 + c[0,2]/4 + c[2,0]/4 - c[2,2]/6, 
                    c[0,1]/4 - c[0,2]/6 - c[2,1]/6 + c[2,2]/4], 
                   [c[1,0]/4 + c[1,2]/6 - c[2,0]/3 - c[2,2]/4, 
                    c[1,1]/6 + c[1,2]/4 - c[2,1]/4 - c[2,2]/3, 
                    -c[1,0]/4 + c[1,2]/3 + c[2,0]/3 - c[2,2]/4, 
                    c[1,1]/3 - c[1,2]/4 - c[2,1]/4 + c[2,2]/3, 
                    -c[1,0]/4 - c[1,2]/3 + c[2,0]/6 + c[2,2]/4, 
                    -c[1,1]/3 - c[1,2]/4 + c[2,1]/4 + c[2,2]/6, 
                    c[1,0]/4 - c[1,2]/6 - c[2,0]/6 + c[2,2]/4, 
                    -c[1,1]/6 + c[1,2]/4 + c[2,1]/4 - c[2,2]/6], 
                   [-c[0,0]/6 - c[0,2]/4 - c[2,0]/4 - c[2,2]/6, 
                    -c[0,1]/4 - c[0,2]/6 - c[2,1]/6 - c[2,2]/4, 
                    c[0,0]/6 - c[0,2]/4 + c[2,0]/4 - c[2,2]/3, 
                    -c[0,1]/4 + c[0,2]/6 - c[2,1]/3 + c[2,2]/4, 
                    c[0,0]/3 + c[0,2]/4 + c[2,0]/4 + c[2,2]/3, 
                    c[0,1]/4 + c[0,2]/3 + c[2,1]/3 + c[2,2]/4, 
                    -c[0,0]/3 + c[0,2]/4 - c[2,0]/4 + c[2,2]/6, 
                    c[0,1]/4 - c[0,2]/3 + c[2,1]/6 - c[2,2]/4], 
                   [-c[1,0]/4 - c[1,2]/6 - c[2,0]/6 - c[2,2]/4, 
                    -c[1,1]/6 - c[1,2]/4 - c[2,1]/4 - c[2,2]/6, 
                    c[1,0]/4 - c[1,2]/3 + c[2,0]/6 - c[2,2]/4, 
                    -c[1,1]/3 + c[1,2]/4 - c[2,1]/4 + c[2,2]/6, 
                    c[1,0]/4 + c[1,2]/3 + c[2,0]/3 + c[2,2]/4, 
                    c[1,1]/3 + c[1,2]/4 + c[2,1]/4 + c[2,2]/3, 
                    -c[1,0]/4 + c[1,2]/6 - c[2,0]/3 + c[2,2]/4, 
                    c[1,1]/6 - c[1,2]/4 + c[2,1]/4 - c[2,2]/3], 
                   [c[0,0]/6 + c[0,2]/4 - c[2,0]/4 - c[2,2]/3, 
                    c[0,1]/4 + c[0,2]/6 - c[2,1]/3 - c[2,2]/4, 
                    -c[0,0]/6 + c[0,2]/4 + c[2,0]/4 - c[2,2]/6, 
                    c[0,1]/4 - c[0,2]/6 - c[2,1]/6 + c[2,2]/4, 
                    -c[0,0]/3 - c[0,2]/4 + c[2,0]/4 + c[2,2]/6, 
                    -c[0,1]/4 - c[0,2]/3 + c[2,1]/6 + c[2,2]/4, 
                    c[0,0]/3 - c[0,2]/4 - c[2,0]/4 + c[2,2]/3, 
                    -c[0,1]/4 + c[0,2]/3 + c[2,1]/3 - c[2,2]/4], 
                   [-c[1,0]/4 - c[1,2]/3 + c[2,0]/6 + c[2,2]/4, 
                    -c[1,1]/3 - c[1,2]/4 + c[2,1]/4 + c[2,2]/6, 
                    c[1,0]/4 - c[1,2]/6 - c[2,0]/6 + c[2,2]/4, 
                    -c[1,1]/6 + c[1,2]/4 + c[2,1]/4 - c[2,2]/6, 
                    c[1,0]/4 + c[1,2]/6 - c[2,0]/3 - c[2,2]/4, 
                    c[1,1]/6 + c[1,2]/4 - c[2,1]/4 - c[2,2]/3, 
                    -c[1,0]/4 + c[1,2]/3 + c[2,0]/3 - c[2,2]/4, 
                    c[1,1]/3 - c[1,2]/4 - c[2,1]/4 + c[2,2]/3]])
    return Ke