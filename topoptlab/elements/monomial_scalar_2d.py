# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import shape_functions, jacobian

def _lm_monomial_2d(xe: np.ndarray, u: np.ndarray, n: int,
                    p: np.ndarray = np.array([1.]),
                    t: np.ndarray = np.array([1.]),
                    quadr_method: str = "gauss-legendre",
                    nquad: int = 3,
                    **kwargs: Any) -> np.ndarray:
    """
    Create element matrix for a monomial of a scalar field in 2D with bilinear
    quadrilateral elements. The special case for polynomial of order 1 is the
    mass matrix

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    u : np.ndarray, shape (nels,4)
        nodal scalar field variable.
    n : int
        polynomial order
    p : np.ndarray of shape (nels) or (1)
        scalar prefactor for each element
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
        element polynomial matrix.

    """
    #
    if len(xe.shape) == 2 and len(u.shape)==1:
        xe,u = xe[None,:,:],u[None,:]
    if len(u.shape) == 1:
        u = u[None,:]
    if xe.shape[0]-1 >= u.shape[0]:
        nel = xe.shape[0]
    else:
        nel = u.shape[0]
    #
    if isinstance(p,float) or (p.shape[0] == 1 and xe.shape[0] !=1):
        p = np.full(nel, p)
    #
    if isinstance(t,float):
        t = np.array([t])
    # shape (ncoords,4*)
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # (nq,4)
    N = shape_functions(xi=xi,eta=eta)
    #
    integral = N[None,:,:,None]@N[None,:,:,None].transpose([0,1,3,2])
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=True)
    detJ = ((J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])).reshape(nel,nq)
    # multiply by determinant and quadrature
    factor = ((u@N.transpose())**(n-1))[:,:,None,None]
    Ke = (w[None,:,None,None]*integral*factor*detJ[:,:,None,None]).sum(axis=1)
    #
    return t[:,None,None] * p[:,None,None] * Ke

def lm_cubic_2d(u: np.ndarray,
                p: float = 1.,
                l: np.ndarray = np.array([1.,1.]),
                t: float = 1.,
                **kwargs: Any) -> np.ndarray:
    """
    Create element matrix for a cubic monomial for scalar field in 2D with
    bilinear quadrilateral Lagrangian elements.

    Parameters
    ----------
    p : float
        scalar prefactor
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    v = l[0]*l[1]*t
    return p*v*np.array([[(72*u[0]**2 + 36*u[0]*u[1] + 9*u[0]*u[2] + 36*u[0]*u[3] + 12*u[1]**2 + 6*u[1]*u[2] + 9*u[1]*u[3] + 2*u[2]**2 + 6*u[2]*u[3] + 12*u[3]**2)/1800,
           (18*u[0]**2 + 24*u[0]*u[1] + 6*u[0]*u[2] + 9*u[0]*u[3] + 18*u[1]**2 + 9*u[1]*u[2] + 6*u[1]*u[3] + 3*u[2]**2 + 4*u[2]*u[3] + 3*u[3]**2)/1800,
           (9*u[0]**2 + 12*u[0]*u[1] + 8*u[0]*u[2] + 12*u[0]*u[3] + 9*u[1]**2 + 12*u[1]*u[2] + 8*u[1]*u[3] + 9*u[2]**2 + 12*u[2]*u[3] + 9*u[3]**2)/3600,
           (18*u[0]**2 + 9*u[0]*u[1] + 6*u[0]*u[2] + 24*u[0]*u[3] + 3*u[1]**2 + 4*u[1]*u[2] + 6*u[1]*u[3] + 3*u[2]**2 + 9*u[2]*u[3] + 18*u[3]**2)/1800],
          [(18*u[0]**2 + 24*u[0]*u[1] + 6*u[0]*u[2] + 9*u[0]*u[3] + 18*u[1]**2 + 9*u[1]*u[2] + 6*u[1]*u[3] + 3*u[2]**2 + 4*u[2]*u[3] + 3*u[3]**2)/1800,
           (12*u[0]**2 + 36*u[0]*u[1] + 9*u[0]*u[2] + 6*u[0]*u[3] + 72*u[1]**2 + 36*u[1]*u[2] + 9*u[1]*u[3] + 12*u[2]**2 + 6*u[2]*u[3] + 2*u[3]**2)/1800,
           (3*u[0]**2 + 9*u[0]*u[1] + 6*u[0]*u[2] + 4*u[0]*u[3] + 18*u[1]**2 + 24*u[1]*u[2] + 6*u[1]*u[3] + 18*u[2]**2 + 9*u[2]*u[3] + 3*u[3]**2)/1800,
           (9*u[0]**2 + 12*u[0]*u[1] + 8*u[0]*u[2] + 12*u[0]*u[3] + 9*u[1]**2 + 12*u[1]*u[2] + 8*u[1]*u[3] + 9*u[2]**2 + 12*u[2]*u[3] + 9*u[3]**2)/3600],
          [(9*u[0]**2 + 12*u[0]*u[1] + 8*u[0]*u[2] + 12*u[0]*u[3] + 9*u[1]**2 + 12*u[1]*u[2] + 8*u[1]*u[3] + 9*u[2]**2 + 12*u[2]*u[3] + 9*u[3]**2)/3600,
           (3*u[0]**2 + 9*u[0]*u[1] + 6*u[0]*u[2] + 4*u[0]*u[3] + 18*u[1]**2 + 24*u[1]*u[2] + 6*u[1]*u[3] + 18*u[2]**2 + 9*u[2]*u[3] + 3*u[3]**2)/1800,
           (2*u[0]**2 + 6*u[0]*u[1] + 9*u[0]*u[2] + 6*u[0]*u[3] + 12*u[1]**2 + 36*u[1]*u[2] + 9*u[1]*u[3] + 72*u[2]**2 + 36*u[2]*u[3] + 12*u[3]**2)/1800,
           (3*u[0]**2 + 4*u[0]*u[1] + 6*u[0]*u[2] + 9*u[0]*u[3] + 3*u[1]**2 + 9*u[1]*u[2] + 6*u[1]*u[3] + 18*u[2]**2 + 24*u[2]*u[3] + 18*u[3]**2)/1800],
          [(18*u[0]**2 + 9*u[0]*u[1] + 6*u[0]*u[2] + 24*u[0]*u[3] + 3*u[1]**2 + 4*u[1]*u[2] + 6*u[1]*u[3] + 3*u[2]**2 + 9*u[2]*u[3] + 18*u[3]**2)/1800,
           (9*u[0]**2 + 12*u[0]*u[1] + 8*u[0]*u[2] + 12*u[0]*u[3] + 9*u[1]**2 + 12*u[1]*u[2] + 8*u[1]*u[3] + 9*u[2]**2 + 12*u[2]*u[3] + 9*u[3]**2)/3600,
           (3*u[0]**2 + 4*u[0]*u[1] + 6*u[0]*u[2] + 9*u[0]*u[3] + 3*u[1]**2 + 9*u[1]*u[2] + 6*u[1]*u[3] + 18*u[2]**2 + 24*u[2]*u[3] + 18*u[3]**2)/1800,
           (12*u[0]**2 + 6*u[0]*u[1] + 9*u[0]*u[2] + 36*u[0]*u[3] + 2*u[1]**2 + 6*u[1]*u[2] + 9*u[1]*u[3] + 12*u[2]**2 + 36*u[2]*u[3] + 72*u[3]**2)/1800]])
