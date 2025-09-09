# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.bilinear_quadrilateral import shape_functions,jacobian
from topoptlab.fem import get_integrpoints

def _lf_bodyforce_2d(xe: np.ndarray,
                     b: np.ndarray = np.array([0,-1.]),
                     t: np.ndarray = np.array([1.]),
                     quadr_method: str = "gauss-legendre",
                     nquad: int = 1,
                     **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on bilinear quadrilateral elements (1st order) due to
    bodyforce (e. g. gravity).

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    b : np.ndarray of shape (nels,2) or (2)
        body force (e. g. density*gravity_acceleration)
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
    fe : np.ndarray, shape (nels,8,1)
        nodal forces.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if (len(b.shape) == 1) or (b.shape[0] == 1):
        b = np.full((xe.shape[0],2), b)
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # shape functions have shape (nq,4)
    N = np.kron(shape_functions(xi=xi, eta=eta)[:,:,None], np.eye(2))
    #
    integral = N[None,:,:,:] @ b[:,None,None,:].transpose(0,1,3,2)
    #integral = b[:,None,None,:] @ N[None,:,:,:].transpose(0,1,3,2)
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=True)
    detJ = ((J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])).reshape(nel,nq)
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return t[:,None,None] * fe

def lf_bodyforce_2d(b: np.ndarray = np.array([0,-1]),
                    l: np.ndarray = np.array([1.,1.]),
                    t: float = 1.,
                    **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on bilinear quadrilateral Lagrangian element
    (1st order) due to bodyforce (e. g. gravity) via analytical integration.
    Element is a parallelogram.

    Parameters
    ----------
    b : np.ndarray shape (2)
        body force (e. g. density*gravity_acceleration)
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    fe : np.ndarray, shape (8,1)
        nodal forces.

    """
    A = l[0]*l[1] / 4
    return t*A*np.array([[b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]]])
