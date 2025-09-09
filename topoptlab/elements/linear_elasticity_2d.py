# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.bilinear_quadrilateral import bmatrix
from topoptlab.fem import get_integrpoints

def _lk_linear_elast_2d(xe: np.ndarray, c: np.ndarray,
                        quadr_method: str = "gauss-legendre",
                        t: np.ndarray = np.array([1.]),
                        nquad: int = 2,
                        **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D linear elasticity with
    bilinear quadrilateral Lagrangian elements.

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

def lk_linear_elast_2d(E: float = 1, nu: float = 0.3,
                       plane_stress: bool = True,
                       l: np.ndarray = np.array([1.,1.]), 
                       g: np.ndarray = np.array([0.]),
                       t: float = 1.,
                       **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D isotropic linear elasticity with
    bilinear quadrilateral Lagrangian elements in plane stress.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson' ratio.
    plane_stress : bool 
        if True, plane stress is assumed, otherwise plane strain.
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
    if plane_stress:
        return t*np.array([[E*(2*l[0]**2*nu - 2*l[0]**2 - 3*l[0]*l[1]*nu*np.tan(g[0]) + 3*l[0]*l[1]*np.tan(g[0]) + 2*l[1]**2*nu*np.tan(g[0])**2 - 2*l[1]**2*np.tan(g[0])**2 - 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2*(nu**2 - 1) + l[1]*(nu + 1)*(3*l[0]*nu*np.tan(g[0]) - 3*l[0]*np.tan(g[0]) - l[1]*nu*np.tan(g[0])**2 + l[1]*np.tan(g[0])**2 + 2*l[1]))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2*nu + 2*l[0]**2 + l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*np.tan(g[0])**2 - 2*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1))],
                           [E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2 + 3*l[0]*l[1]*np.tan(g[0]) + l[1]**2*nu + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(2*l[0]**2 + l[1]*(-6*l[0]*np.tan(g[0]) - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(4*l[0]**2 + l[1]**2*nu + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(nu**2 - 1))],
                           [E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(2*l[0]**2*(nu**2 - 1) + 3*l[0]*l[1]*(nu**2 - 1)*np.tan(g[0]) + 2*l[1]**2*(nu + 1)*(nu*np.tan(g[0])**2 - np.tan(g[0])**2 - 2))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2*(nu**2 - 1) + l[1]**2*(nu + 1)*(nu*np.tan(g[0])**2 - np.tan(g[0])**2 - 2))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2*(nu**2 - 1) + l[1]*(nu + 1)*(-3*l[0]*nu*np.tan(g[0]) + 3*l[0]*np.tan(g[0]) - l[1]*nu*np.tan(g[0])**2 + l[1]*np.tan(g[0])**2 + 2*l[1]))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1))],
                           [E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2 - 3*l[0]*l[1]*np.tan(g[0]) + l[1]**2*(nu - 2*np.tan(g[0])**2 - 1))/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(4*l[0]**2 + l[1]**2*(nu - 2*np.tan(g[0])**2 - 1))/(12*l[0]*l[1]*(nu**2 - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(2*l[0]**2 + l[1]*(6*l[0]*np.tan(g[0]) - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(nu**2 - 1))],
                           [E*(-l[0]**2*(nu**2 - 1) + l[1]*(nu + 1)*(3*l[0]*nu*np.tan(g[0]) - 3*l[0]*np.tan(g[0]) - l[1]*nu*np.tan(g[0])**2 + l[1]*np.tan(g[0])**2 + 2*l[1]))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2*(nu**2 - 1) + l[1]**2*(nu + 1)*(nu*np.tan(g[0])**2 - np.tan(g[0])**2 - 2))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(2*l[0]**2*(nu**2 - 1) - 3*l[0]*l[1]*(nu**2 - 1)*np.tan(g[0]) + 2*l[1]**2*(nu + 1)*(nu*np.tan(g[0])**2 - np.tan(g[0])**2 - 2))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1))],
                           [E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(2*l[0]**2 + l[1]*(-6*l[0]*np.tan(g[0]) - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(4*l[0]**2 + l[1]**2*(nu - 2*np.tan(g[0])**2 - 1))/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2 + 3*l[0]*l[1]*np.tan(g[0]) + l[1]**2*(nu - 2*np.tan(g[0])**2 - 1))/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(nu**2 - 1))],
                           [E*(-2*l[0]**2*nu + 2*l[0]**2 + l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*np.tan(g[0])**2 - 2*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2*(nu**2 - 1) + l[1]*(nu + 1)*(-3*l[0]*nu*np.tan(g[0]) + 3*l[0]*np.tan(g[0]) - l[1]*nu*np.tan(g[0])**2 + l[1]*np.tan(g[0])**2 + 2*l[1]))/(12*l[0]*l[1]*(nu + 1)*(nu**2 - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(2*l[0]**2*nu - 2*l[0]**2 + 3*l[0]*l[1]*nu*np.tan(g[0]) - 3*l[0]*l[1]*np.tan(g[0]) + 2*l[1]**2*nu*np.tan(g[0])**2 - 2*l[1]**2*np.tan(g[0])**2 - 4*l[1]**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1))],
                           [E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(4*l[0]**2 + l[1]**2*nu + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(nu**2 - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(2*l[0]**2 + l[1]*(6*l[0]*np.tan(g[0]) - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(nu**2 - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu**2 - 1)),
                            E*(-l[0]**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(nu**2 - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(nu - 1)),
                            E*(-2*l[0]**2 - 3*l[0]*l[1]*np.tan(g[0]) + l[1]**2*nu + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(6*l[0]*l[1]*(nu**2 - 1))]])
    else:
        return t*np.array([[E*(2*l[0]**2*nu - 2*l[0]**2 - 3*l[0]*l[1]*nu*np.tan(g[0]) + 3*l[0]*l[1]*np.tan(g[0]) + 2*l[1]**2*nu*np.tan(g[0])**2 + 4*l[1]**2*nu - 2*l[1]**2*np.tan(g[0])**2 - 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - 4*l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(3*l[0]**2*(1 - nu) + 2*l[0]**2*(nu - 1) + l[1]*(3*l[0]*nu*np.tan(g[0]) - 3*l[0]*np.tan(g[0]) - l[1]*nu - l[1]*nu/np.cos(g[0])**2 + l[1] + l[1]/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(-2*l[0]**2*nu + 2*l[0]**2 + l[1]**2*nu + l[1]**2*nu/np.cos(g[0])**2 - l[1]**2 - l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1))],
                           [E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(2*l[0]**2*nu - 2*l[0]**2 - 3*l[0]*l[1]*nu*np.tan(g[0]) + 3*l[0]*l[1]*np.tan(g[0]) - l[1]**2*nu + 2*l[1]**2*nu/np.cos(g[0])**2 + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(6*l[0]**2*(1 - nu) + 4*l[0]**2*(nu - 1) + l[1]*(6*l[0]*nu*np.tan(g[0]) - 6*l[0]*np.tan(g[0]) - 2*l[1]*nu*np.tan(g[0])**2 - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(-4*l[0]**2*nu + 4*l[0]**2 - l[1]**2*nu + 2*l[1]**2*nu/np.cos(g[0])**2 + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1))],
                           [E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - 4*l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(2*l[0]**2*(nu - 1) + 3*l[0]*l[1]*(nu - 1)*np.tan(g[0]) + 2*l[1]**2*(nu + nu/np.cos(g[0])**2 - 1 - 1/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(2*l[0]**2*(1 - nu) + l[1]**2*(nu + nu/np.cos(g[0])**2 - 1 - 1/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(3*l[0]**2*(1 - nu) + 2*l[0]**2*(nu - 1) + l[1]*(-3*l[0]*nu*np.tan(g[0]) + 3*l[0]*np.tan(g[0]) - l[1]*nu - l[1]*nu/np.cos(g[0])**2 + l[1] + l[1]/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1))],
                           [E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(2*l[0]**2*(nu - 1) + 3*l[0]*l[1]*(nu - 1)*np.tan(g[0]) + l[1]**2*(2*nu*np.tan(g[0])**2 + nu - 2*np.tan(g[0])**2 - 1))/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(4*l[0]**2*(1 - nu) + l[1]**2*(2*nu*np.tan(g[0])**2 + nu - 2*np.tan(g[0])**2 - 1))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(6*l[0]**2*(1 - nu) + 4*l[0]**2*(nu - 1) + l[1]*(-6*l[0]*nu*np.tan(g[0]) + 6*l[0]*np.tan(g[0]) - 2*l[1]*nu*np.tan(g[0])**2 - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(2*nu**2 + nu - 1))],
                           [E*(3*l[0]**2*(1 - nu) + 2*l[0]**2*(nu - 1) + l[1]*(3*l[0]*nu*np.tan(g[0]) - 3*l[0]*np.tan(g[0]) - l[1]*nu - l[1]*nu/np.cos(g[0])**2 + l[1] + l[1]/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                             E*(2*l[0]**2*(1 - nu) + l[1]**2*(nu + nu/np.cos(g[0])**2 - 1 - 1/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(2*l[0]**2*(nu - 1) + 3*l[0]*l[1]*(1 - nu)*np.tan(g[0]) + 2*l[1]**2*(nu + nu/np.cos(g[0])**2 - 1 - 1/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - 4*l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1))],
                           [E*(3*l[0] - 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(6*l[0]**2*(1 - nu) + 4*l[0]**2*(nu - 1) + l[1]*(6*l[0]*nu*np.tan(g[0]) - 6*l[0]*np.tan(g[0]) - 2*l[1]*nu*np.tan(g[0])**2 - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(4*l[0]**2*(1 - nu) + l[1]**2*(2*nu*np.tan(g[0])**2 + nu - 2*np.tan(g[0])**2 - 1))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(2*l[0]**2*(nu - 1) + 3*l[0]*l[1]*(1 - nu)*np.tan(g[0]) + l[1]**2*(2*nu*np.tan(g[0])**2 + nu - 2*np.tan(g[0])**2 - 1))/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1))],
                           [E*(-2*l[0]**2*nu + 2*l[0]**2 + l[1]**2*nu + l[1]**2*nu/np.cos(g[0])**2 - l[1]**2 - l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(3*l[0]**2*(1 - nu) + 2*l[0]**2*(nu - 1) + l[1]*(-3*l[0]*nu*np.tan(g[0]) + 3*l[0]*np.tan(g[0]) - l[1]*nu - l[1]*nu/np.cos(g[0])**2 + l[1] + l[1]/np.cos(g[0])**2))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - 4*l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(9*l[0]*nu - 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(2*l[0]**2*nu - 2*l[0]**2 + 3*l[0]*l[1]*nu*np.tan(g[0]) - 3*l[0]*l[1]*np.tan(g[0]) + 2*l[1]**2*nu*np.tan(g[0])**2 + 4*l[1]**2*nu - 2*l[1]**2*np.tan(g[0])**2 - 4*l[1]**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1))],
                           [E*(9*l[0]*nu - 3*l[0] + 2*l[1]*nu*np.tan(g[0]) + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(-4*l[0]**2*nu + 4*l[0]**2 - l[1]**2*nu + 2*l[1]**2*nu/np.cos(g[0])**2 + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                           -E*(3*l[0] + 2*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(6*l[0]**2*(1 - nu) + 4*l[0]**2*(nu - 1) + l[1]*(-6*l[0]*nu*np.tan(g[0]) + 6*l[0]*np.tan(g[0]) - 2*l[1]*nu*np.tan(g[0])**2 - l[1]*nu + 2*l[1]*np.tan(g[0])**2 + l[1]))/(12*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(-9*l[0]*nu + 3*l[0] - 4*l[1]*nu*np.tan(g[0]) - 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu**2 + nu - 1)),
                            E*(l[0]**2*nu - l[0]**2 - 2*l[1]**2*nu*np.tan(g[0])**2 - l[1]**2*nu + 2*l[1]**2*np.tan(g[0])**2 + l[1]**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1)),
                            E*(3*l[0] + 4*l[1]*np.tan(g[0]))/(24*l[0]*(2*nu - 1)),
                            E*(2*l[0]**2*nu - 2*l[0]**2 + 3*l[0]*l[1]*nu*np.tan(g[0]) - 3*l[0]*l[1]*np.tan(g[0]) - l[1]**2*nu + 2*l[1]**2*nu/np.cos(g[0])**2 + l[1]**2 - 2*l[1]**2/np.cos(g[0])**2)/(6*l[0]*l[1]*(2*nu**2 + nu - 1))]])

def lk_linear_elast_aniso_2d(c: np.ndarray,
                             l: np.ndarray = np.array([1.,1.]), 
                             g: np.ndarray = np.array([0.]),
                             t: float = 1.,
                             **kwargs: Any) -> np.ndarray:
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
    return t*np.array([[c[0,0]*l[1]/(3*l[0]) + c[0,2]/4 - c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,0]/4 - c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(3*l[1]) - c[2,2]*np.tan(g[0])/2 + c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       c[0,1]/4 - c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(3*l[1]) - c[2,1]*np.tan(g[0])/2 + c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       -c[0,0]*l[1]/(3*l[0]) + c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(6*l[1]) - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       -c[0,0]*l[1]/(6*l[0]) - c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/2 - c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       -c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(6*l[1]) + c[2,1]*np.tan(g[0])/2 - c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       c[0,0]*l[1]/(6*l[0]) - c[0,2]/4 - c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,0]/4 - c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(3*l[1]) + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       -c[0,1]/4 - c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(6*l[0])],
                      [c[1,0]/4 - c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(3*l[1]) - c[1,2]*np.tan(g[0])/2 + c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(3*l[0]) + c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(3*l[1]) - c[1,1]*np.tan(g[0])/2 + c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[1,2]/4 - c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,1]/4 - c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[1]/(3*l[0]),
                       -c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(6*l[1]) - c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,0]*l[1]/(3*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(6*l[1]) - c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,2]*l[1]/(3*l[0]),
                       -c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/2 - c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(6*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(6*l[1]) + c[1,1]*np.tan(g[0])/2 - c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[1]/(6*l[0]),
                       c[1,0]/4 - c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(3*l[1]) + c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[2,0]*l[1]/(6*l[0]) - c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(3*l[1]) + c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[1,2]/4 - c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,1]/4 - c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,2]*l[1]/(6*l[0])],
                      [-c[0,0]*l[1]/(3*l[0]) - c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(6*l[1]) - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       -c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       (4*c[2,2]*l[0]**2 + 3*l[0]*l[1]*(-c[0,2] - c[2,0] + 2*c[2,2]*np.tan(g[0])) + 4*l[1]**2*(c[0,0] - c[0,2]*np.tan(g[0]) - c[2,0]*np.tan(g[0]) + c[2,2]*np.tan(g[0])**2))/(12*l[0]*l[1]),
                       (4*c[2,1]*l[0]**2 + 3*l[0]*l[1]*(-c[0,1] + 2*c[2,1]*np.tan(g[0]) - c[2,2]) + 4*l[1]**2*(-c[0,1]*np.tan(g[0]) + c[0,2] + c[2,1]*np.tan(g[0])**2 - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (-4*c[2,2]*l[0]**2 + 3*l[0]*l[1]*(c[0,2] - c[2,0]) + 2*l[1]**2*(c[0,0] - c[0,2]*np.tan(g[0]) - c[2,0]*np.tan(g[0]) + c[2,2]*np.tan(g[0])**2))/(12*l[0]*l[1]),
                       (-4*c[2,1]*l[0]**2 + 3*l[0]*l[1]*(c[0,1] - c[2,2]) + 2*l[1]**2*(-c[0,1]*np.tan(g[0]) + c[0,2] + c[2,1]*np.tan(g[0])**2 - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       -c[0,0]*l[1]/(6*l[0]) + c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/2 - c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(6*l[1]) - c[2,1]*np.tan(g[0])/2 - c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0])],
                      [c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(6*l[1]) - c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(6*l[1]) - c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,2]*l[1]/(3*l[0]),
                       (4*c[1,2]*l[0]**2 + 3*l[0]*l[1]*(-c[1,0] + 2*c[1,2]*np.tan(g[0]) - c[2,2]) + 4*l[1]**2*(-c[1,0]*np.tan(g[0]) + c[1,2]*np.tan(g[0])**2 + c[2,0] - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (4*c[1,1]*l[0]**2 + 3*l[0]*l[1]*(2*c[1,1]*np.tan(g[0]) - c[1,2] - c[2,1]) + 4*l[1]**2*(c[1,1]*np.tan(g[0])**2 - c[1,2]*np.tan(g[0]) - c[2,1]*np.tan(g[0]) + c[2,2]))/(12*l[0]*l[1]),
                       (-4*c[1,2]*l[0]**2 + 3*l[0]*l[1]*(-c[1,0] + c[2,2]) + 2*l[1]**2*(-c[1,0]*np.tan(g[0]) + c[1,2]*np.tan(g[0])**2 + c[2,0] - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (-4*c[1,1]*l[0]**2 + 3*l[0]*l[1]*(-c[1,2] + c[2,1]) + 2*l[1]**2*(c[1,1]*np.tan(g[0])**2 - c[1,2]*np.tan(g[0]) - c[2,1]*np.tan(g[0]) + c[2,2]))/(12*l[0]*l[1]),
                       c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/2 - c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(6*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(6*l[1]) - c[1,1]*np.tan(g[0])/2 - c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[1]/(6*l[0])],
                      [-c[0,0]*l[1]/(6*l[0]) - c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(6*l[1]) + c[2,2]*np.tan(g[0])/2 - c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       -c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(6*l[1]) + c[2,1]*np.tan(g[0])/2 - c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       (-4*c[2,2]*l[0]**2 + 3*l[0]*l[1]*(-c[0,2] + c[2,0]) + 2*l[1]**2*(c[0,0] - c[0,2]*np.tan(g[0]) - c[2,0]*np.tan(g[0]) + c[2,2]*np.tan(g[0])**2))/(12*l[0]*l[1]),
                       (-4*c[2,1]*l[0]**2 + 3*l[0]*l[1]*(-c[0,1] + c[2,2]) + 2*l[1]**2*(-c[0,1]*np.tan(g[0]) + c[0,2] + c[2,1]*np.tan(g[0])**2 - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (4*c[2,2]*l[0]**2 + 3*l[0]*l[1]*(c[0,2] + c[2,0] - 2*c[2,2]*np.tan(g[0])) + 4*l[1]**2*(c[0,0] - c[0,2]*np.tan(g[0]) - c[2,0]*np.tan(g[0]) + c[2,2]*np.tan(g[0])**2))/(12*l[0]*l[1]),
                       (4*c[2,1]*l[0]**2 + 3*l[0]*l[1]*(c[0,1] - 2*c[2,1]*np.tan(g[0]) + c[2,2]) + 4*l[1]**2*(-c[0,1]*np.tan(g[0]) + c[0,2] + c[2,1]*np.tan(g[0])**2 - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       -c[0,0]*l[1]/(3*l[0]) + c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(6*l[1]) - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0])],
                      [-c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(6*l[1]) + c[1,2]*np.tan(g[0])/2 - c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(6*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(6*l[1]) + c[1,1]*np.tan(g[0])/2 - c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[1]/(6*l[0]),
                       (-4*c[1,2]*l[0]**2 + 3*l[0]*l[1]*(c[1,0] - c[2,2]) + 2*l[1]**2*(-c[1,0]*np.tan(g[0]) + c[1,2]*np.tan(g[0])**2 + c[2,0] - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (-4*c[1,1]*l[0]**2 + 3*l[0]*l[1]*(c[1,2] - c[2,1]) + 2*l[1]**2*(c[1,1]*np.tan(g[0])**2 - c[1,2]*np.tan(g[0]) - c[2,1]*np.tan(g[0]) + c[2,2]))/(12*l[0]*l[1]),
                       (4*c[1,2]*l[0]**2 + 3*l[0]*l[1]*(c[1,0] - 2*c[1,2]*np.tan(g[0]) + c[2,2]) + 4*l[1]**2*(-c[1,0]*np.tan(g[0]) + c[1,2]*np.tan(g[0])**2 + c[2,0] - c[2,2]*np.tan(g[0])))/(12*l[0]*l[1]),
                       (4*c[1,1]*l[0]**2 + 3*l[0]*l[1]*(-2*c[1,1]*np.tan(g[0]) + c[1,2] + c[2,1]) + 4*l[1]**2*(c[1,1]*np.tan(g[0])**2 - c[1,2]*np.tan(g[0]) - c[2,1]*np.tan(g[0]) + c[2,2]))/(12*l[0]*l[1]),
                       -c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(6*l[1]) - c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,0]*l[1]/(3*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(6*l[1]) - c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,2]*l[1]/(3*l[0])],
                      [c[0,0]*l[1]/(6*l[0]) + c[0,2]/4 - c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,0]/4 - c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(3*l[1]) + c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       c[0,1]/4 - c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(3*l[1]) + c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[0,0]*l[1]/(6*l[0]) + c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[0]/(6*l[1]) - c[2,2]*np.tan(g[0])/2 - c[2,2]*l[1]*np.tan(g[0])**2/(6*l[0]),
                       c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[0,2]*l[1]/(6*l[0]) - c[2,1]*l[0]/(6*l[1]) - c[2,1]*np.tan(g[0])/2 - c[2,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[0,0]*l[1]/(3*l[0]) - c[0,2]/4 + c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,0]/4 + c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(6*l[1]) - c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       -c[0,1]/4 + c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(6*l[1]) - c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[0,0]*l[1]/(3*l[0]) - c[0,2]/4 - c[0,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,0]/4 - c[2,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[0]/(3*l[1]) + c[2,2]*np.tan(g[0])/2 + c[2,2]*l[1]*np.tan(g[0])**2/(3*l[0]),
                       -c[0,1]/4 - c[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + c[0,2]*l[1]/(3*l[0]) + c[2,1]*l[0]/(3*l[1]) + c[2,1]*np.tan(g[0])/2 + c[2,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(3*l[0])],
                      [-c[1,0]/4 - c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(3*l[1]) + c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[2,0]*l[1]/(6*l[0]) + c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(3*l[1]) + c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[1,2]/4 - c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,1]/4 - c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,2]*l[1]/(6*l[0]),
                       c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - c[1,2]*l[0]/(6*l[1]) - c[1,2]*np.tan(g[0])/2 - c[1,2]*l[1]*np.tan(g[0])**2/(6*l[0]) - c[2,0]*l[1]/(6*l[0]) + c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(6*l[0]),
                       -c[1,1]*l[0]/(6*l[1]) - c[1,1]*np.tan(g[0])/2 - c[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]) + c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(6*l[0]) + c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(6*l[0]) - c[2,2]*l[1]/(6*l[0]),
                       c[1,0]/4 + c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(6*l[1]) - c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[2,0]*l[1]/(3*l[0]) - c[2,2]/4 + c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(6*l[1]) - c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[1,2]/4 + c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,1]/4 + c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,2]*l[1]/(3*l[0]),
                       -c[1,0]/4 - c[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + c[1,2]*l[0]/(3*l[1]) + c[1,2]*np.tan(g[0])/2 + c[1,2]*l[1]*np.tan(g[0])**2/(3*l[0]) + c[2,0]*l[1]/(3*l[0]) - c[2,2]/4 - c[2,2]*l[1]*np.tan(g[0])/(3*l[0]),
                       c[1,1]*l[0]/(3*l[1]) + c[1,1]*np.tan(g[0])/2 + c[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]) - c[1,2]/4 - c[1,2]*l[1]*np.tan(g[0])/(3*l[0]) - c[2,1]/4 - c[2,1]*l[1]*np.tan(g[0])/(3*l[0]) + c[2,2]*l[1]/(3*l[0])]])

def _lf_strain_2d(xe: np.ndarray, eps: np.ndarray, c: np.ndarray,
                  quadr_method: str = "gauss-legendre",
                  t: np.ndarray = np.array([1.]),
                  nquad: int = 2,
                  **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on bilinear quadrilateral Lagrangian element
    (1st order) due to a uniform strain via numerical integration.
    We assume anisotropic elasticity.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    eps : np.ndarray shape (nels,3) or (3)
        uniform strain in Voigt notation.
    c : np.ndarray, shape (3,3)
        stiffness tensor in Voigt notation.
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
    fe : np.ndarray, shape (nels,8,1)
        nodal forces.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if (len(eps.shape) == 1) or (eps.shape[0] == 1):
        eps = np.full((xe.shape[0],3), eps)
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
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]@eps[:,None,None,:].transpose(0,1,3,2)
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    # multiply thickness
    return t[:,None,None] * fe

def lf_strain_2d(eps, E=1,nu=0.3,
                 plane_stress=True,
                 l=np.array([1.,1.]), g = np.array([0.]),
                 t=1.,
                 **kwargs):
    """
    Compute nodal forces on bilinear quadrilateral Lagrangian element
    (1st order) due to a uniform strain via analytical integration.
    Element shape is a parallelogram and we assume isotropic elasticity.

    Parameters
    ----------
    eps : np.ndarray shape (3)
        uniform strain in Voigt notation.
    E : float
        Young's modulus.
    nu : float
        Poisson' ratio.
    plane_stress : bool 
        if True, plane stress is assumed, otherwise plane strain.
    l : np.ndarray (2)
        side length of element
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element

    Returns
    -------
    fe : np.ndarray, shape (8,1)
        nodal forces.

    """
    if plane_stress:
        return t*np.array([[E*(2*eps[0]*l[1] + 2*eps[1]*l[1]*nu - eps[2]*l[0]*nu + eps[2]*l[0] + eps[2]*l[1]*nu*np.tan(g[0]) - eps[2]*l[1]*np.tan(g[0]))/(4*(nu**2 - 1))],
                          [E*(2*eps[0]*l[0]*nu - 2*eps[0]*l[1]*nu*np.tan(g[0]) + 2*eps[1]*l[0] - 2*eps[1]*l[1]*np.tan(g[0]) - eps[2]*l[1]*nu + eps[2]*l[1])/(4*(nu**2 - 1))],
                          [E*(-2*eps[0]*l[1] - 2*eps[1]*l[1]*nu - eps[2]*l[0]*nu + eps[2]*l[0] - eps[2]*l[1]*nu*np.tan(g[0]) + eps[2]*l[1]*np.tan(g[0]))/(4*(nu**2 - 1))],
                          [E*(2*l[0]*(eps[0]*nu + eps[1]) + l[1]*(2*eps[0]*nu*np.tan(g[0]) + 2*eps[1]*np.tan(g[0]) + eps[2]*nu - eps[2]))/(4*(nu**2 - 1))],
                          [E*(-2*eps[0]*l[1] - 2*eps[1]*l[1]*nu + eps[2]*l[0]*nu - eps[2]*l[0] - eps[2]*l[1]*nu*np.tan(g[0]) + eps[2]*l[1]*np.tan(g[0]))/(4*(nu**2 - 1))],
                          [E*(-2*l[0]*(eps[0]*nu + eps[1]) + l[1]*(2*eps[0]*nu*np.tan(g[0]) + 2*eps[1]*np.tan(g[0]) + eps[2]*nu - eps[2]))/(4*(nu**2 - 1))],
                          [E*(2*eps[0]*l[1] + 2*eps[1]*l[1]*nu + eps[2]*l[0]*nu - eps[2]*l[0] + eps[2]*l[1]*nu*np.tan(g[0]) - eps[2]*l[1]*np.tan(g[0]))/(4*(nu**2 - 1))],
                          [E*(-2*eps[0]*l[0]*nu - 2*eps[0]*l[1]*nu*np.tan(g[0]) - 2*eps[1]*l[0] - 2*eps[1]*l[1]*np.tan(g[0]) - eps[2]*l[1]*nu + eps[2]*l[1])/(4*(nu**2 - 1))]])
    else:
        return t*np.array([[E*(-2*eps[0]*l[1]*nu + 2*eps[0]*l[1] + 2*eps[1]*l[1]*nu - eps[2]*l[0]*nu + eps[2]*l[0] + eps[2]*l[1]*nu*np.tan(g[0]) - eps[2]*l[1]*np.tan(g[0]))/(4*(2*nu**2 + nu - 1))],
                          [E*(2*eps[0]*l[0]*nu - 2*eps[0]*l[1]*nu*np.tan(g[0]) - 2*eps[1]*l[0]*nu + 2*eps[1]*l[0] + 2*eps[1]*l[1]*nu*np.tan(g[0]) - 2*eps[1]*l[1]*np.tan(g[0]) - eps[2]*l[1]*nu + eps[2]*l[1])/(4*(2*nu**2 + nu - 1))],
                          [E*(2*eps[0]*l[1]*nu - 2*eps[0]*l[1] - 2*eps[1]*l[1]*nu - eps[2]*l[0]*nu + eps[2]*l[0] - eps[2]*l[1]*nu*np.tan(g[0]) + eps[2]*l[1]*np.tan(g[0]))/(4*(2*nu**2 + nu - 1))],
                          [E*(2*l[0]*(eps[0]*nu - eps[1]*nu + eps[1]) + l[1]*(2*eps[0]*nu*np.tan(g[0]) - 2*eps[1]*nu*np.tan(g[0]) + 2*eps[1]*np.tan(g[0]) + eps[2]*nu - eps[2]))/(4*(2*nu**2 + nu - 1))],
                          [E*(2*eps[0]*l[1]*nu - 2*eps[0]*l[1] - 2*eps[1]*l[1]*nu + eps[2]*l[0]*nu - eps[2]*l[0] - eps[2]*l[1]*nu*np.tan(g[0]) + eps[2]*l[1]*np.tan(g[0]))/(4*(2*nu**2 + nu - 1))],
                          [E*(2*l[0]*(-eps[0]*nu + eps[1]*nu - eps[1]) + l[1]*(2*eps[0]*nu*np.tan(g[0]) - 2*eps[1]*nu*np.tan(g[0]) + 2*eps[1]*np.tan(g[0]) + eps[2]*nu - eps[2]))/(4*(2*nu**2 + nu - 1))],
                          [E*(-2*eps[0]*l[1]*nu + 2*eps[0]*l[1] + 2*eps[1]*l[1]*nu + eps[2]*l[0]*nu - eps[2]*l[0] + eps[2]*l[1]*nu*np.tan(g[0]) - eps[2]*l[1]*np.tan(g[0]))/(4*(2*nu**2 + nu - 1))],
                          [E*(-2*eps[0]*l[0]*nu - 2*eps[0]*l[1]*nu*np.tan(g[0]) + 2*eps[1]*l[0]*nu - 2*eps[1]*l[0] + 2*eps[1]*l[1]*nu*np.tan(g[0]) - 2*eps[1]*l[1]*np.tan(g[0]) - eps[2]*l[1]*nu + eps[2]*l[1])/(4*(2*nu**2 + nu - 1))]])


def lf_strain_aniso_2d(eps: np.ndarray, c: np.ndarray,
                       l: np.ndarray = np.array([1.,1.]), 
                       g: np.ndarray = np.array([0.]),
                       t: float = 1.,
                       **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on bilinear quadrilateral Lagrangian element
    (1st order) due to a uniform strain via analytical integration.
    Element shape is a parallelogram and we assume anisotropic elasticity.

    Parameters
    ----------
    eps : np.ndarray shape (3)
        uniform strain in Voigt notation.
    c : np.ndarray, shape (3,3)
        stiffness tensor in Voigt notation.
    l : np.ndarray (2)
        side length of element
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element

    Returns
    -------
    fe : np.ndarray, shape (8,1)
        nodal forces.

    """
    return t*np.array([[-c[0,0]*eps[0]*l[1]/2 - c[0,1]*eps[1]*l[1]/2 - c[0,2]*eps[2]*l[1]/2 - c[2,0]*eps[0]*l[0]/2 + c[2,0]*eps[0]*l[1]*np.tan(g[0])/2 - c[2,1]*eps[1]*l[0]/2 + c[2,1]*eps[1]*l[1]*np.tan(g[0])/2 - c[2,2]*eps[2]*l[0]/2 + c[2,2]*eps[2]*l[1]*np.tan(g[0])/2],
                      [-c[1,0]*eps[0]*l[0]/2 + c[1,0]*eps[0]*l[1]*np.tan(g[0])/2 - c[1,1]*eps[1]*l[0]/2 + c[1,1]*eps[1]*l[1]*np.tan(g[0])/2 - c[1,2]*eps[2]*l[0]/2 + c[1,2]*eps[2]*l[1]*np.tan(g[0])/2 - c[2,0]*eps[0]*l[1]/2 - c[2,1]*eps[1]*l[1]/2 - c[2,2]*eps[2]*l[1]/2],
                      [c[0,0]*eps[0]*l[1]/2 + c[0,1]*eps[1]*l[1]/2 + c[0,2]*eps[2]*l[1]/2 - c[2,0]*eps[0]*l[0]/2 - c[2,0]*eps[0]*l[1]*np.tan(g[0])/2 - c[2,1]*eps[1]*l[0]/2 - c[2,1]*eps[1]*l[1]*np.tan(g[0])/2 - c[2,2]*eps[2]*l[0]/2 - c[2,2]*eps[2]*l[1]*np.tan(g[0])/2],
                      [-c[1,0]*eps[0]*l[0]/2 - c[1,0]*eps[0]*l[1]*np.tan(g[0])/2 - c[1,1]*eps[1]*l[0]/2 - c[1,1]*eps[1]*l[1]*np.tan(g[0])/2 - c[1,2]*eps[2]*l[0]/2 - c[1,2]*eps[2]*l[1]*np.tan(g[0])/2 + c[2,0]*eps[0]*l[1]/2 + c[2,1]*eps[1]*l[1]/2 + c[2,2]*eps[2]*l[1]/2],
                      [c[0,0]*eps[0]*l[1]/2 + c[0,1]*eps[1]*l[1]/2 + c[0,2]*eps[2]*l[1]/2 + c[2,0]*eps[0]*l[0]/2 - c[2,0]*eps[0]*l[1]*np.tan(g[0])/2 + c[2,1]*eps[1]*l[0]/2 - c[2,1]*eps[1]*l[1]*np.tan(g[0])/2 + c[2,2]*eps[2]*l[0]/2 - c[2,2]*eps[2]*l[1]*np.tan(g[0])/2],
                      [c[1,0]*eps[0]*l[0]/2 - c[1,0]*eps[0]*l[1]*np.tan(g[0])/2 + c[1,1]*eps[1]*l[0]/2 - c[1,1]*eps[1]*l[1]*np.tan(g[0])/2 + c[1,2]*eps[2]*l[0]/2 - c[1,2]*eps[2]*l[1]*np.tan(g[0])/2 + c[2,0]*eps[0]*l[1]/2 + c[2,1]*eps[1]*l[1]/2 + c[2,2]*eps[2]*l[1]/2],
                      [-c[0,0]*eps[0]*l[1]/2 - c[0,1]*eps[1]*l[1]/2 - c[0,2]*eps[2]*l[1]/2 + c[2,0]*eps[0]*l[0]/2 + c[2,0]*eps[0]*l[1]*np.tan(g[0])/2 + c[2,1]*eps[1]*l[0]/2 + c[2,1]*eps[1]*l[1]*np.tan(g[0])/2 + c[2,2]*eps[2]*l[0]/2 + c[2,2]*eps[2]*l[1]*np.tan(g[0])/2],
                      [c[1,0]*eps[0]*l[0]/2 + c[1,0]*eps[0]*l[1]*np.tan(g[0])/2 + c[1,1]*eps[1]*l[0]/2 + c[1,1]*eps[1]*l[1]*np.tan(g[0])/2 + c[1,2]*eps[2]*l[0]/2 + c[1,2]*eps[2]*l[1]*np.tan(g[0])/2 - c[2,0]*eps[0]*l[1]/2 - c[2,1]*eps[1]*l[1]/2 - c[2,2]*eps[2]*l[1]/2]])