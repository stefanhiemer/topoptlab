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
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # 
    B,detJ = bmatrix(xi=xi, eta=eta, xe=xe, 
                     all_elems=True, 
                     return_detJ=True)
    detJ = detJ.reshape(nel,nq)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]@B
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    # multiply thickness
    return t[:,None,None] * Ke

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

def lk_linear_elast_aniso_2d(c,
                             l=np.array([1.,1.]), g = [0.],
                             t=1.):
    """
    Create element stiffness matrix for 2D anisotropic linear elasticity with 
    bilinear quadrilateral elements. 
    
    Parameters
    ----------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    return t * np.array([[-c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[0,2]*np.tan(g[0])/3 + c[0,2]/4 - c[2,0]*np.tan(g[0])/3 + c[2,0]/4 + c[2,2]*l[0]/(3*l[1]),
                          -c[0,1]*np.tan(g[0])/3 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,2]*l[1]/(3*l[0]*np.cos(g[0])**2) + c[2,1]*l[0]/(3*l[1]) - c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,0]*l[1]/(3*l[0]) - c[0,2]*np.tan(g[0])/6 + c[0,2]/4 - c[2,0]*np.tan(g[0])/6 - c[2,0]/4 + c[2,2]*l[0]/(6*l[1]),
                          -c[0,1]*np.tan(g[0])/6 + c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[0,2]*np.tan(g[0])/6 - c[0,2]/4 + c[2,0]*np.tan(g[0])/6 - c[2,0]/4 - c[2,2]*l[0]/(6*l[1]),
                          c[0,1]*np.tan(g[0])/6 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,2]*l[1]/(6*l[0]*np.cos(g[0])**2) - c[2,1]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,0]*l[1]/(6*l[0]) + c[0,2]*np.tan(g[0])/3 - c[0,2]/4 + c[2,0]*np.tan(g[0])/3 + c[2,0]/4 - c[2,2]*l[0]/(3*l[1]),
                          c[0,1]*np.tan(g[0])/3 - c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,2]*np.tan(g[0])/3 + c[2,2]/4],
                         [-c[1,0]*np.tan(g[0])/3 + c[1,0]/4 + c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          c[1,1]*l[0]/(3*l[1]) - c[1,2]*np.tan(g[0])/3 + c[1,2]/4 - c[2,1]*np.tan(g[0])/3 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,2]*l[1]/(3*l[0]*np.cos(g[0])**2),
                          -c[1,0]*np.tan(g[0])/6 - c[1,0]/4 + c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          c[1,1]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/6 - c[1,2]/4 - c[2,1]*np.tan(g[0])/6 + c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]*l[1]/(3*l[0]),
                          c[1,0]*np.tan(g[0])/6 - c[1,0]/4 - c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          -c[1,1]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/6 - c[1,2]/4 + c[2,1]*np.tan(g[0])/6 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,2]*l[1]/(6*l[0]*np.cos(g[0])**2),
                          c[1,0]*np.tan(g[0])/3 + c[1,0]/4 - c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(6*l[0]) + c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          -c[1,1]*l[0]/(3*l[1]) + c[1,2]*np.tan(g[0])/3 + c[1,2]/4 + c[2,1]*np.tan(g[0])/3 - c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]*l[1]/(6*l[0])],
                         [c[0,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,0]*l[1]/(3*l[0]) - c[0,2]*np.tan(g[0])/6 - c[0,2]/4 - c[2,0]*np.tan(g[0])/6 + c[2,0]/4 + c[2,2]*l[0]/(6*l[1]),
                          -c[0,1]*np.tan(g[0])/6 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[0,2]*np.tan(g[0])/3 - c[0,2]/4 - c[2,0]*np.tan(g[0])/3 - c[2,0]/4 + c[2,2]*l[0]/(3*l[1]),
                          -c[0,1]*np.tan(g[0])/3 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,2]*l[1]/(3*l[0]*np.cos(g[0])**2) + c[2,1]*l[0]/(3*l[1]) - c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,0]*l[1]/(6*l[0]) + c[0,2]*np.tan(g[0])/3 + c[0,2]/4 + c[2,0]*np.tan(g[0])/3 - c[2,0]/4 - c[2,2]*l[0]/(3*l[1]),
                          c[0,1]*np.tan(g[0])/3 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[0,2]*np.tan(g[0])/6 + c[0,2]/4 + c[2,0]*np.tan(g[0])/6 + c[2,0]/4 - c[2,2]*l[0]/(6*l[1]),
                          c[0,1]*np.tan(g[0])/6 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,2]*l[1]/(6*l[0]*np.cos(g[0])**2) - c[2,1]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/6 + c[2,2]/4],
                         [-c[1,0]*np.tan(g[0])/6 + c[1,0]/4 + c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          c[1,1]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/6 + c[1,2]/4 - c[2,1]*np.tan(g[0])/6 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]*l[1]/(3*l[0]),
                          -c[1,0]*np.tan(g[0])/3 - c[1,0]/4 + c[1,2]*l[0]/(3*l[1]) + c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          c[1,1]*l[0]/(3*l[1]) - c[1,2]*np.tan(g[0])/3 - c[1,2]/4 - c[2,1]*np.tan(g[0])/3 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,2]*l[1]/(3*l[0]*np.cos(g[0])**2),
                          c[1,0]*np.tan(g[0])/3 - c[1,0]/4 - c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(6*l[0]) + c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          -c[1,1]*l[0]/(3*l[1]) + c[1,2]*np.tan(g[0])/3 - c[1,2]/4 + c[2,1]*np.tan(g[0])/3 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]*l[1]/(6*l[0]),
                          c[1,0]*np.tan(g[0])/6 + c[1,0]/4 - c[1,2]*l[0]/(6*l[1]) - c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          -c[1,1]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/6 + c[1,2]/4 + c[2,1]*np.tan(g[0])/6 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,2]*l[1]/(6*l[0]*np.cos(g[0])**2)],
                         [c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[0,2]*np.tan(g[0])/6 - c[0,2]/4 + c[2,0]*np.tan(g[0])/6 - c[2,0]/4 - c[2,2]*l[0]/(6*l[1]),
                          c[0,1]*np.tan(g[0])/6 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,2]*l[1]/(6*l[0]*np.cos(g[0])**2) - c[2,1]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,0]*l[1]/(6*l[0]) + c[0,2]*np.tan(g[0])/3 - c[0,2]/4 + c[2,0]*np.tan(g[0])/3 + c[2,0]/4 - c[2,2]*l[0]/(3*l[1]),
                          c[0,1]*np.tan(g[0])/3 - c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[0,2]*np.tan(g[0])/3 + c[0,2]/4 - c[2,0]*np.tan(g[0])/3 + c[2,0]/4 + c[2,2]*l[0]/(3*l[1]),
                          -c[0,1]*np.tan(g[0])/3 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,2]*l[1]/(3*l[0]*np.cos(g[0])**2) + c[2,1]*l[0]/(3*l[1]) - c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,0]*l[1]/(3*l[0]) - c[0,2]*np.tan(g[0])/6 + c[0,2]/4 - c[2,0]*np.tan(g[0])/6 - c[2,0]/4 + c[2,2]*l[0]/(6*l[1]),
                          -c[0,1]*np.tan(g[0])/6 + c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/6 - c[2,2]/4],
                         [c[1,0]*np.tan(g[0])/6 - c[1,0]/4 - c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          -c[1,1]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/6 - c[1,2]/4 + c[2,1]*np.tan(g[0])/6 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,2]*l[1]/(6*l[0]*np.cos(g[0])**2),
                          c[1,0]*np.tan(g[0])/3 + c[1,0]/4 - c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(6*l[0]) + c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          -c[1,1]*l[0]/(3*l[1]) + c[1,2]*np.tan(g[0])/3 + c[1,2]/4 + c[2,1]*np.tan(g[0])/3 - c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]*l[1]/(6*l[0]),
                          -c[1,0]*np.tan(g[0])/3 + c[1,0]/4 + c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          c[1,1]*l[0]/(3*l[1]) - c[1,2]*np.tan(g[0])/3 + c[1,2]/4 - c[2,1]*np.tan(g[0])/3 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,2]*l[1]/(3*l[0]*np.cos(g[0])**2),
                          -c[1,0]*np.tan(g[0])/6 - c[1,0]/4 + c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          c[1,1]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/6 - c[1,2]/4 - c[2,1]*np.tan(g[0])/6 + c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]*l[1]/(3*l[0])],
                         [-c[0,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,0]*l[1]/(6*l[0]) + c[0,2]*np.tan(g[0])/3 + c[0,2]/4 + c[2,0]*np.tan(g[0])/3 - c[2,0]/4 - c[2,2]*l[0]/(3*l[1]),
                          c[0,1]*np.tan(g[0])/3 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          -c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[0,2]*np.tan(g[0])/6 + c[0,2]/4 + c[2,0]*np.tan(g[0])/6 + c[2,0]/4 - c[2,2]*l[0]/(6*l[1]),
                          c[0,1]*np.tan(g[0])/6 + c[0,1]/4 - c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[0,2]*l[1]/(6*l[0]*np.cos(g[0])**2) - c[2,1]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,0]*l[1]/(3*l[0]) - c[0,2]*np.tan(g[0])/6 - c[0,2]/4 - c[2,0]*np.tan(g[0])/6 + c[2,0]/4 + c[2,2]*l[0]/(6*l[1]),
                          -c[0,1]*np.tan(g[0])/6 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          c[0,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[0,2]*np.tan(g[0])/3 - c[0,2]/4 - c[2,0]*np.tan(g[0])/3 - c[2,0]/4 + c[2,2]*l[0]/(3*l[1]),
                          -c[0,1]*np.tan(g[0])/3 - c[0,1]/4 + c[0,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[0,2]*l[1]/(3*l[0]*np.cos(g[0])**2) + c[2,1]*l[0]/(3*l[1]) - c[2,2]*np.tan(g[0])/3 - c[2,2]/4],
                         [c[1,0]*np.tan(g[0])/3 - c[1,0]/4 - c[1,2]*l[0]/(3*l[1]) - c[2,0]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(6*l[0]) + c[2,2]*np.tan(g[0])/3 + c[2,2]/4,
                          -c[1,1]*l[0]/(3*l[1]) + c[1,2]*np.tan(g[0])/3 - c[1,2]/4 + c[2,1]*np.tan(g[0])/3 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]*l[1]/(6*l[0]),
                          c[1,0]*np.tan(g[0])/6 + c[1,0]/4 - c[1,2]*l[0]/(6*l[1]) - c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,0]*l[1]/(6*l[0]*np.cos(g[0])**2) + c[2,2]*np.tan(g[0])/6 + c[2,2]/4,
                          -c[1,1]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/6 + c[1,2]/4 + c[2,1]*np.tan(g[0])/6 + c[2,1]/4 - c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) - c[2,2]*l[1]/(6*l[0]*np.cos(g[0])**2),
                          -c[1,0]*np.tan(g[0])/6 + c[1,0]/4 + c[1,2]*l[0]/(6*l[1]) + c[2,0]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]*np.tan(g[0])/6 - c[2,2]/4,
                          c[1,1]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/6 + c[1,2]/4 - c[2,1]*np.tan(g[0])/6 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]*l[1]/(3*l[0]),
                          -c[1,0]*np.tan(g[0])/3 - c[1,0]/4 + c[1,2]*l[0]/(3*l[1]) + c[2,0]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,0]*l[1]/(3*l[0]*np.cos(g[0])**2) - c[2,2]*np.tan(g[0])/3 - c[2,2]/4,
                          c[1,1]*l[0]/(3*l[1]) - c[1,2]*np.tan(g[0])/3 - c[1,2]/4 - c[2,1]*np.tan(g[0])/3 - c[2,1]/4 + c[2,2]*l[1]*np.tan(g[0])/(2*l[0]) + c[2,2]*l[1]/(3*l[0]*np.cos(g[0])**2)]]) 
