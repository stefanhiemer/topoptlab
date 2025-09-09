# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

import numpy as np

from topoptlab.elements.bilinear_quadrilateral import shape_functions,bmatrix
from topoptlab.fem import get_integrpoints

def _fk_heatexp_2d(xe: np.ndarray, c: np.ndarray,
                   a: np.ndarray, DeltaT: Union[None,np.ndarray] = None,
                   t: np.ndarray = np.array([1.]),
                   quadr_method: str = "gauss-legendre",
                   nquad: int = 2,
                   **kwargs: Any) -> np.ndarray:
    """
    Create force vector for 2D heat expansion with
    bilinear quadrilateral Lagrangian elements. This amounts to

    int_Omega B_T @ C_v @ alpha_v @ N_T @ dOmega DeltaT

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    c : np.ndarray, shape (nels,3,3) or (3,3)
        stiffness tensor.
    a : np.ndarray, shape (nels,2,2) or (2,2)
        linear heat expansion tensor.
    DeltaT : np.ndarray shape (nels,4) or None
        difference of nodal temperatures with respect to reference temperature.
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
    ft : np.ndarray, shape (nels,8) or shape (nels,8,4)
        nodal forces due to thermal expansion.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    # convert linear heat expansion tensor to Voigt notation (nel,3,1)
    if len(a.shape) == 2:
        a = a[None,:,:]
    a = a[:,[0,1,0],[0,1,1],None]
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    #
    if DeltaT is not None:
        if len(DeltaT.shape) == 1:
            DeltaT = DeltaT[None,:]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # shape functions at integration points
    N = shape_functions(xi=xi,eta=eta)[None,:,:,None]
    #
    B,detJ = bmatrix(xi=xi, eta=eta, xe=xe,
                     all_elems=True,
                     return_detJ=True)
    detJ = detJ.reshape(nel,nq)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]
    integral = integral@a[:,None,:,:]@N.transpose([0,1,3,2])
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    if DeltaT is None:
        return t[:,None,None] * fe
    else:
        # this is basically a matrix product
        return np.sum(t[:,None,None] * fe * DeltaT[:,None,:],axis=2)

def fk_heatexp_2d(E: float, nu: float,
                  a: float, DeltaT: Union[None,np.ndarray] = None,
                  l: np.ndarray = np.array([1.,1.]), 
                  g: np.ndarray = np.array([0.]),
                  t = np.array([1.]),
                  **kwargs: Any) -> np.ndarray:
    """
    Create force vector for 2D heat expansion with
    bilinear quadrilateral Lagrangian elements with plane stress. This amounts to

    int_Omega B_T @ C_v @ alpha_v @ N_T @ dOmega DeltaT

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    a : float
        linear heat expansion coefficient.
    DeltaT : np.ndarray shape (4) or None
        difference of nodal temperatures with respect to reference temperature.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.
    t : np.ndarray of shape (nels) or (1)
        thickness of element

    Returns
    -------
    fe : np.ndarray, shape (8) or (8,4)
        nodal forces due to thermal expansion or matrix that returns forces by
        fe@DeltaT.

    """
    fe = np.array([[E*a*l[1]/(6*(nu - 1)), E*a*l[1]/(6*(nu - 1)), E*a*l[1]/(12*(nu - 1)), E*a*l[1]/(12*(nu - 1))],
          [E*a*(l[0] - l[1]*np.tan(g[0]))/(6*(nu - 1)), E*a*(l[0] - 2*l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(l[0] - l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(2*l[0] - l[1]*np.tan(g[0]))/(12*(nu - 1))],
          [-E*a*l[1]/(6*nu - 6), -E*a*l[1]/(6*nu - 6), -E*a*l[1]/(12*nu - 12), -E*a*l[1]/(12*nu - 12)],
          [E*a*(l[0] + 2*l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(l[0] + l[1]*np.tan(g[0]))/(6*(nu - 1)), E*a*(2*l[0] + l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(l[0] + l[1]*np.tan(g[0]))/(12*(nu - 1))],
          [-E*a*l[1]/(12*nu - 12), -E*a*l[1]/(12*nu - 12), -E*a*l[1]/(6*nu - 6), -E*a*l[1]/(6*nu - 6)],
          [E*a*(-l[0] + l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(-2*l[0] + l[1]*np.tan(g[0]))/(12*(nu - 1)), E*a*(-l[0] + l[1]*np.tan(g[0]))/(6*(nu - 1)), E*a*(-l[0] + 2*l[1]*np.tan(g[0]))/(12*(nu - 1))],
          [E*a*l[1]/(12*(nu - 1)), E*a*l[1]/(12*(nu - 1)), E*a*l[1]/(6*(nu - 1)), E*a*l[1]/(6*(nu - 1))],
          [-E*a*(2*l[0] + l[1]*np.tan(g[0]))/(12*nu - 12), -E*a*(l[0] + l[1]*np.tan(g[0]))/(12*nu - 12), -E*a*(l[0] + 2*l[1]*np.tan(g[0]))/(12*nu - 12), -E*a*(l[0] + l[1]*np.tan(g[0]))/(6*nu - 6)]])
    if DeltaT is None:
        return fe*t
    else:
        return t*fe@DeltaT

def fk_heatexp_aniso_2d(c: np.ndarray,
                        a: np.ndarray, 
                        DeltaT: Union[None,np.ndarray] = None,
                        l: np.ndarray = np.array([1.,1.]), 
                        g: np.ndarray = np.array([0.]),
                        t = np.array([1.]),
                        **kwargs: Any) -> np.ndarray:
    """
    Create force vector for 2D heat expansion with
    bilinear quadrilateral Lagrangian elements. This amounts to

    int_Omega B_T @ C_v @ alpha_v @ N_T @ dOmega DeltaT

    Parameters
    ----------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    a : np.ndarray, shape (2,2)
        anisotropic linear heat expansion tensor. If isotropic a would be
        [[a,0],[0,a]]
    DeltaT : np.ndarray shape (4) or None
        difference of nodal temperatures with respect to reference temperature.
    g : np.ndarray (1)
        angle of parallelogram.
    t : np.ndarray of shape (nels) or (1)
        thickness of element

    Returns
    -------
    fe : np.ndarray, shape (8) or (8,4)
        nodal forces due to thermal expansion or matrix that returns forces by
        fe@DeltaT.

    """
    # convert heat expans. coeff. to Voigt notation
    a = a[[0,1,0],[0,1,1]]
    #
    fe = np.array([[-a[0]*c[0,0]*l[1]/6 - a[0]*c[2,0]*l[0]/6 + a[0]*c[2,0]*l[1]*np.tan(g[0])/6 - a[1]*c[0,1]*l[1]/6 - a[1]*c[2,1]*l[0]/6 + a[1]*c[2,1]*l[1]*np.tan(g[0])/6 - a[2]*c[0,2]*l[1]/6 - a[2]*c[2,2]*l[0]/6 + a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           -a[0]*c[0,0]*l[1]/6 - a[0]*c[2,0]*l[0]/12 + a[0]*c[2,0]*l[1]*np.tan(g[0])/6 - a[1]*c[0,1]*l[1]/6 - a[1]*c[2,1]*l[0]/12 + a[1]*c[2,1]*l[1]*np.tan(g[0])/6 - a[2]*c[0,2]*l[1]/6 - a[2]*c[2,2]*l[0]/12 + a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           -a[0]*c[0,0]*l[1]/12 - a[0]*c[2,0]*l[0]/12 + a[0]*c[2,0]*l[1]*np.tan(g[0])/12 - a[1]*c[0,1]*l[1]/12 - a[1]*c[2,1]*l[0]/12 + a[1]*c[2,1]*l[1]*np.tan(g[0])/12 - a[2]*c[0,2]*l[1]/12 - a[2]*c[2,2]*l[0]/12 + a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           -a[0]*c[0,0]*l[1]/12 - a[0]*c[2,0]*l[0]/6 + a[0]*c[2,0]*l[1]*np.tan(g[0])/12 - a[1]*c[0,1]*l[1]/12 - a[1]*c[2,1]*l[0]/6 + a[1]*c[2,1]*l[1]*np.tan(g[0])/12 - a[2]*c[0,2]*l[1]/12 - a[2]*c[2,2]*l[0]/6 + a[2]*c[2,2]*l[1]*np.tan(g[0])/12],
          [-a[0]*c[1,0]*l[0]/6 + a[0]*c[1,0]*l[1]*np.tan(g[0])/6 - a[0]*c[2,0]*l[1]/6 - a[1]*c[1,1]*l[0]/6 + a[1]*c[1,1]*l[1]*np.tan(g[0])/6 - a[1]*c[2,1]*l[1]/6 - a[2]*c[1,2]*l[0]/6 + a[2]*c[1,2]*l[1]*np.tan(g[0])/6 - a[2]*c[2,2]*l[1]/6,
           -a[0]*c[1,0]*l[0]/12 + a[0]*c[1,0]*l[1]*np.tan(g[0])/6 - a[0]*c[2,0]*l[1]/6 - a[1]*c[1,1]*l[0]/12 + a[1]*c[1,1]*l[1]*np.tan(g[0])/6 - a[1]*c[2,1]*l[1]/6 - a[2]*c[1,2]*l[0]/12 + a[2]*c[1,2]*l[1]*np.tan(g[0])/6 - a[2]*c[2,2]*l[1]/6,
           -a[0]*c[1,0]*l[0]/12 + a[0]*c[1,0]*l[1]*np.tan(g[0])/12 - a[0]*c[2,0]*l[1]/12 - a[1]*c[1,1]*l[0]/12 + a[1]*c[1,1]*l[1]*np.tan(g[0])/12 - a[1]*c[2,1]*l[1]/12 - a[2]*c[1,2]*l[0]/12 + a[2]*c[1,2]*l[1]*np.tan(g[0])/12 - a[2]*c[2,2]*l[1]/12,
           -a[0]*c[1,0]*l[0]/6 + a[0]*c[1,0]*l[1]*np.tan(g[0])/12 - a[0]*c[2,0]*l[1]/12 - a[1]*c[1,1]*l[0]/6 + a[1]*c[1,1]*l[1]*np.tan(g[0])/12 - a[1]*c[2,1]*l[1]/12 - a[2]*c[1,2]*l[0]/6 + a[2]*c[1,2]*l[1]*np.tan(g[0])/12 - a[2]*c[2,2]*l[1]/12],
          [a[0]*c[0,0]*l[1]/6 - a[0]*c[2,0]*l[0]/12 - a[0]*c[2,0]*l[1]*np.tan(g[0])/6 + a[1]*c[0,1]*l[1]/6 - a[1]*c[2,1]*l[0]/12 - a[1]*c[2,1]*l[1]*np.tan(g[0])/6 + a[2]*c[0,2]*l[1]/6 - a[2]*c[2,2]*l[0]/12 - a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           a[0]*c[0,0]*l[1]/6 - a[0]*c[2,0]*l[0]/6 - a[0]*c[2,0]*l[1]*np.tan(g[0])/6 + a[1]*c[0,1]*l[1]/6 - a[1]*c[2,1]*l[0]/6 - a[1]*c[2,1]*l[1]*np.tan(g[0])/6 + a[2]*c[0,2]*l[1]/6 - a[2]*c[2,2]*l[0]/6 - a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           a[0]*c[0,0]*l[1]/12 - a[0]*c[2,0]*l[0]/6 - a[0]*c[2,0]*l[1]*np.tan(g[0])/12 + a[1]*c[0,1]*l[1]/12 - a[1]*c[2,1]*l[0]/6 - a[1]*c[2,1]*l[1]*np.tan(g[0])/12 + a[2]*c[0,2]*l[1]/12 - a[2]*c[2,2]*l[0]/6 - a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           a[0]*c[0,0]*l[1]/12 - a[0]*c[2,0]*l[0]/12 - a[0]*c[2,0]*l[1]*np.tan(g[0])/12 + a[1]*c[0,1]*l[1]/12 - a[1]*c[2,1]*l[0]/12 - a[1]*c[2,1]*l[1]*np.tan(g[0])/12 + a[2]*c[0,2]*l[1]/12 - a[2]*c[2,2]*l[0]/12 - a[2]*c[2,2]*l[1]*np.tan(g[0])/12],
          [-a[0]*c[1,0]*l[0]/12 - a[0]*c[1,0]*l[1]*np.tan(g[0])/6 + a[0]*c[2,0]*l[1]/6 - a[1]*c[1,1]*l[0]/12 - a[1]*c[1,1]*l[1]*np.tan(g[0])/6 + a[1]*c[2,1]*l[1]/6 - a[2]*c[1,2]*l[0]/12 - a[2]*c[1,2]*l[1]*np.tan(g[0])/6 + a[2]*c[2,2]*l[1]/6,
           -a[0]*c[1,0]*l[0]/6 - a[0]*c[1,0]*l[1]*np.tan(g[0])/6 + a[0]*c[2,0]*l[1]/6 - a[1]*c[1,1]*l[0]/6 - a[1]*c[1,1]*l[1]*np.tan(g[0])/6 + a[1]*c[2,1]*l[1]/6 - a[2]*c[1,2]*l[0]/6 - a[2]*c[1,2]*l[1]*np.tan(g[0])/6 + a[2]*c[2,2]*l[1]/6,
           -a[0]*c[1,0]*l[0]/6 - a[0]*c[1,0]*l[1]*np.tan(g[0])/12 + a[0]*c[2,0]*l[1]/12 - a[1]*c[1,1]*l[0]/6 - a[1]*c[1,1]*l[1]*np.tan(g[0])/12 + a[1]*c[2,1]*l[1]/12 - a[2]*c[1,2]*l[0]/6 - a[2]*c[1,2]*l[1]*np.tan(g[0])/12 + a[2]*c[2,2]*l[1]/12,
           -a[0]*c[1,0]*l[0]/12 - a[0]*c[1,0]*l[1]*np.tan(g[0])/12 + a[0]*c[2,0]*l[1]/12 - a[1]*c[1,1]*l[0]/12 - a[1]*c[1,1]*l[1]*np.tan(g[0])/12 + a[1]*c[2,1]*l[1]/12 - a[2]*c[1,2]*l[0]/12 - a[2]*c[1,2]*l[1]*np.tan(g[0])/12 + a[2]*c[2,2]*l[1]/12],
          [a[0]*c[0,0]*l[1]/12 + a[0]*c[2,0]*l[0]/12 - a[0]*c[2,0]*l[1]*np.tan(g[0])/12 + a[1]*c[0,1]*l[1]/12 + a[1]*c[2,1]*l[0]/12 - a[1]*c[2,1]*l[1]*np.tan(g[0])/12 + a[2]*c[0,2]*l[1]/12 + a[2]*c[2,2]*l[0]/12 - a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           a[0]*c[0,0]*l[1]/12 + a[0]*c[2,0]*l[0]/6 - a[0]*c[2,0]*l[1]*np.tan(g[0])/12 + a[1]*c[0,1]*l[1]/12 + a[1]*c[2,1]*l[0]/6 - a[1]*c[2,1]*l[1]*np.tan(g[0])/12 + a[2]*c[0,2]*l[1]/12 + a[2]*c[2,2]*l[0]/6 - a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           a[0]*c[0,0]*l[1]/6 + a[0]*c[2,0]*l[0]/6 - a[0]*c[2,0]*l[1]*np.tan(g[0])/6 + a[1]*c[0,1]*l[1]/6 + a[1]*c[2,1]*l[0]/6 - a[1]*c[2,1]*l[1]*np.tan(g[0])/6 + a[2]*c[0,2]*l[1]/6 + a[2]*c[2,2]*l[0]/6 - a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           a[0]*c[0,0]*l[1]/6 + a[0]*c[2,0]*l[0]/12 - a[0]*c[2,0]*l[1]*np.tan(g[0])/6 + a[1]*c[0,1]*l[1]/6 + a[1]*c[2,1]*l[0]/12 - a[1]*c[2,1]*l[1]*np.tan(g[0])/6 + a[2]*c[0,2]*l[1]/6 + a[2]*c[2,2]*l[0]/12 - a[2]*c[2,2]*l[1]*np.tan(g[0])/6],
          [a[0]*c[1,0]*l[0]/12 - a[0]*c[1,0]*l[1]*np.tan(g[0])/12 + a[0]*c[2,0]*l[1]/12 + a[1]*c[1,1]*l[0]/12 - a[1]*c[1,1]*l[1]*np.tan(g[0])/12 + a[1]*c[2,1]*l[1]/12 + a[2]*c[1,2]*l[0]/12 - a[2]*c[1,2]*l[1]*np.tan(g[0])/12 + a[2]*c[2,2]*l[1]/12,
           a[0]*c[1,0]*l[0]/6 - a[0]*c[1,0]*l[1]*np.tan(g[0])/12 + a[0]*c[2,0]*l[1]/12 + a[1]*c[1,1]*l[0]/6 - a[1]*c[1,1]*l[1]*np.tan(g[0])/12 + a[1]*c[2,1]*l[1]/12 + a[2]*c[1,2]*l[0]/6 - a[2]*c[1,2]*l[1]*np.tan(g[0])/12 + a[2]*c[2,2]*l[1]/12,
           a[0]*c[1,0]*l[0]/6 - a[0]*c[1,0]*l[1]*np.tan(g[0])/6 + a[0]*c[2,0]*l[1]/6 + a[1]*c[1,1]*l[0]/6 - a[1]*c[1,1]*l[1]*np.tan(g[0])/6 + a[1]*c[2,1]*l[1]/6 + a[2]*c[1,2]*l[0]/6 - a[2]*c[1,2]*l[1]*np.tan(g[0])/6 + a[2]*c[2,2]*l[1]/6,
           a[0]*c[1,0]*l[0]/12 - a[0]*c[1,0]*l[1]*np.tan(g[0])/6 + a[0]*c[2,0]*l[1]/6 + a[1]*c[1,1]*l[0]/12 - a[1]*c[1,1]*l[1]*np.tan(g[0])/6 + a[1]*c[2,1]*l[1]/6 + a[2]*c[1,2]*l[0]/12 - a[2]*c[1,2]*l[1]*np.tan(g[0])/6 + a[2]*c[2,2]*l[1]/6],
          [-a[0]*c[0,0]*l[1]/12 + a[0]*c[2,0]*l[0]/6 + a[0]*c[2,0]*l[1]*np.tan(g[0])/12 - a[1]*c[0,1]*l[1]/12 + a[1]*c[2,1]*l[0]/6 + a[1]*c[2,1]*l[1]*np.tan(g[0])/12 - a[2]*c[0,2]*l[1]/12 + a[2]*c[2,2]*l[0]/6 + a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           -a[0]*c[0,0]*l[1]/12 + a[0]*c[2,0]*l[0]/12 + a[0]*c[2,0]*l[1]*np.tan(g[0])/12 - a[1]*c[0,1]*l[1]/12 + a[1]*c[2,1]*l[0]/12 + a[1]*c[2,1]*l[1]*np.tan(g[0])/12 - a[2]*c[0,2]*l[1]/12 + a[2]*c[2,2]*l[0]/12 + a[2]*c[2,2]*l[1]*np.tan(g[0])/12,
           -a[0]*c[0,0]*l[1]/6 + a[0]*c[2,0]*l[0]/12 + a[0]*c[2,0]*l[1]*np.tan(g[0])/6 - a[1]*c[0,1]*l[1]/6 + a[1]*c[2,1]*l[0]/12 + a[1]*c[2,1]*l[1]*np.tan(g[0])/6 - a[2]*c[0,2]*l[1]/6 + a[2]*c[2,2]*l[0]/12 + a[2]*c[2,2]*l[1]*np.tan(g[0])/6,
           -a[0]*c[0,0]*l[1]/6 + a[0]*c[2,0]*l[0]/6 + a[0]*c[2,0]*l[1]*np.tan(g[0])/6 - a[1]*c[0,1]*l[1]/6 + a[1]*c[2,1]*l[0]/6 + a[1]*c[2,1]*l[1]*np.tan(g[0])/6 - a[2]*c[0,2]*l[1]/6 + a[2]*c[2,2]*l[0]/6 + a[2]*c[2,2]*l[1]*np.tan(g[0])/6],
          [a[0]*c[1,0]*l[0]/6 + a[0]*c[1,0]*l[1]*np.tan(g[0])/12 - a[0]*c[2,0]*l[1]/12 + a[1]*c[1,1]*l[0]/6 + a[1]*c[1,1]*l[1]*np.tan(g[0])/12 - a[1]*c[2,1]*l[1]/12 + a[2]*c[1,2]*l[0]/6 + a[2]*c[1,2]*l[1]*np.tan(g[0])/12 - a[2]*c[2,2]*l[1]/12,
           a[0]*c[1,0]*l[0]/12 + a[0]*c[1,0]*l[1]*np.tan(g[0])/12 - a[0]*c[2,0]*l[1]/12 + a[1]*c[1,1]*l[0]/12 + a[1]*c[1,1]*l[1]*np.tan(g[0])/12 - a[1]*c[2,1]*l[1]/12 + a[2]*c[1,2]*l[0]/12 + a[2]*c[1,2]*l[1]*np.tan(g[0])/12 - a[2]*c[2,2]*l[1]/12,
           a[0]*c[1,0]*l[0]/12 + a[0]*c[1,0]*l[1]*np.tan(g[0])/6 - a[0]*c[2,0]*l[1]/6 + a[1]*c[1,1]*l[0]/12 + a[1]*c[1,1]*l[1]*np.tan(g[0])/6 - a[1]*c[2,1]*l[1]/6 + a[2]*c[1,2]*l[0]/12 + a[2]*c[1,2]*l[1]*np.tan(g[0])/6 - a[2]*c[2,2]*l[1]/6,
           a[0]*c[1,0]*l[0]/6 + a[0]*c[1,0]*l[1]*np.tan(g[0])/6 - a[0]*c[2,0]*l[1]/6 + a[1]*c[1,1]*l[0]/6 + a[1]*c[1,1]*l[1]*np.tan(g[0])/6 - a[1]*c[2,1]*l[1]/6 + a[2]*c[1,2]*l[0]/6 + a[2]*c[1,2]*l[1]*np.tan(g[0])/6 - a[2]*c[2,2]*l[1]/6]])
    if DeltaT is None:
        return fe*t
    else:
        return t*fe@DeltaT
