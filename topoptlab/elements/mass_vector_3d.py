from typing import Any 

import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.trilinear_hexahedron import jacobian,shape_functions

def _lm_mass_3d(xe: np.ndarray ,p: np.ndarray = np.array([1.]),
                quadr_method: str = "gauss-legendre",
                nquad: int = 2,
                **kwargs: Any) -> np.ndarray:
    """
    Create element mass matrix for vector field in 3D with trilinear hexahedral
    elements.

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
    Ke : np.ndarray, shape (nels,24,24)
        element mass matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if isinstance(p,float) or (p.shape[0] == 1 and xe.shape[0] !=1):
        p = np.full(xe.shape[0], p)
    #
    x,w=get_integrpoints(ndim=3,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, 3,axis=1)]
    #
    N = np.kron(shape_functions(xi=xi,eta=eta,zeta=zeta)[:,:,None], np.eye(3))
    #
    integral = N[None,:,:,:]@N[None,:,:,:].transpose([0,1,3,2])
    # calculate determinant of jacobiann
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=True)
    detJ = (J[:,0,0]*(J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1])-
            J[:,0,1]*(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])+
            J[:,0,2]*(J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0])).reshape(nel,nq)
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return p[:,None,None] * Ke

def lm_mass_3d(p: float = 1.0,
               l: np.ndarray = np.array([1.,1.,1.])):
    """
    Create element mass matrix for vector field in 3D with trilinear hexahedral
    elements.

    Parameters
    ----------
    p : float
        density of element
    l : np.ndarray (3)
        side length of element

    Returns
    -------
    Ke : np.ndarray, shape (24,24)
        element stiffness matrix.

    """
    v = l[0]*l[1]*l[2]
    return p*v*np.array([[1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0],
                          [0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0],
                          [0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108],
                          [1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0],
                          [0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0],
                          [0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216],
                          [1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0],
                          [0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0],
                          [0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108],
                          [1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0],
                          [0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0],
                          [0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54],
                          [1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0],
                          [0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0],
                          [0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54],
                          [1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0, 0],
                          [0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108, 0],
                          [0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0, 1/108],
                          [1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0, 0],
                          [0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54, 0],
                          [0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0, 1/54],
                          [1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0, 0],
                          [0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27, 0],
                          [0, 0, 1/108, 0, 0, 1/216, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/54, 0, 0, 1/108, 0, 0, 1/54, 0, 0, 1/27]])
