# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import shape_functions, jacobian

def _lm_mass_2d(xe: np.ndarray,
                p: np.ndarray = np.array([1.]),
                t: np.ndarray = np.array([1.]),
                quadr_method: str = "gauss-legendre",
                nquad: int = 2,
                **kwargs: Any) -> np.ndarray:
    """
    Create element mass matrix for vector field in 2D with bilinear
    quadrilateral elements.

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
    Ke : np.ndarray, shape (nels,8,8)
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
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    N = np.kron(shape_functions(xi=xi, eta=eta)[:,:,None], np.eye(2))
    #
    integral = N[None,:,:,:]@N[None,:,:,:].transpose([0,1,3,2])
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=True)
    detJ = ((J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])).reshape(nel,nq)
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return t[:,None,None] * p[:,None,None] * Ke

def lm_mass_2d(p: float = 1.,
               l: np.ndarray = np.array([1.,1.]),
               t: float = 1.,
               **kwargs: Any) -> np.ndarray:
    """
    Create mass matrix for vector field in 2D with bilinear quadrilateral
    Lagrangian elements.

    Parameters
    ----------
    p : float
        density of element
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.

    """
    v = l[0]*l[1]*t
    return p*v*np.array([[1/9, 0, 1/18, 0, 1/36, 0, 1/18, 0],
                         [0, 1/9, 0, 1/18, 0, 1/36, 0, 1/18],
                         [1/18, 0, 1/9, 0, 1/18, 0, 1/36, 0],
                         [0, 1/18, 0, 1/9, 0, 1/18, 0, 1/36],
                         [1/36, 0, 1/18, 0, 1/9, 0, 1/18, 0],
                         [0, 1/36, 0, 1/18, 0, 1/9, 0, 1/18],
                         [1/18, 0, 1/36, 0, 1/18, 0, 1/9, 0],
                         [0, 1/18, 0, 1/36, 0, 1/18, 0, 1/9]])
