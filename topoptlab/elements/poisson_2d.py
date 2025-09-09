# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.bilinear_quadrilateral import invjacobian,shape_functions_dxi

def _lk_poisson_2d(xe: np.ndarray, k: np.ndarray,
                   quadr_method: str ="gauss-legendre",
                   t: np.ndarray = np.array([1.]),
                   nquad: int = 2,
                   **kwargs: Any) -> np.ndarray:
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
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if len(k.shape) == 2:
        k = k[None,:,:]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    #
    Jinv,detJ = invjacobian(xi=xi,eta=eta,xe=xe,
                            all_elems=True,return_det=True)
    Jinv = Jinv.reshape(nel,nq,2,2)
    detJ = detJ.reshape(nel,nq)
    gradN = shape_functions_dxi(xi=xi,eta=eta)[None,:,:,:]@Jinv.transpose((0,1,3,2))
    #
    integral = gradN@k[:,None,:,:]@gradN.transpose([0,1,3,2])
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return t[:,None,None] * Ke

def lk_poisson_2d(k: float = 1.,
                  l: np.ndarray = np.array([1.,1.]), 
                  g: np.ndarray = np.array([0.]),
                  t: float = 1.) -> np.ndarray:
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadrilateral elements.

    Parameters
    ----------
    k : float
        heat conductivity.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element.

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    return t*k*np.array([[l[0]/(3*l[1]) - np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) + np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2)],
                         [l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) + np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) - np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2)],
                         [-l[0]/(6*l[1]) + np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) - np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2)],
                         [-l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) - np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) + np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2)]])

def lk_poisson_aniso_2d(k: np.ndarray,
                        l: np.ndarray = np.array([1.,1.]), 
                        g: np.ndarray = np.array([0.]),
                        t: float = 1.) -> np.ndarray:
    """
    Create element stiffness matrix for anisotropic 2D Poisson with bilinear
    quadrilateral elements.

    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        anisotropic heat conductivity. If isotropic k would be [[k,0],[0,k]]
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    return t*np.array([[k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(3*l[1]) - k[1,1]*np.tan(g[0])/2 + k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           -k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           -k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) + k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(3*l[1]) + k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0])],
          [-k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           (4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(-k[0,1] - k[1,0] + 2*k[1,1]*np.tan(g[0])) + 4*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           (-4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(k[0,1] - k[1,0]) + 2*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           -k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) - k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0])],
          [-k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) + k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           (-4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(-k[0,1] + k[1,0]) + 2*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           (4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(k[0,1] + k[1,0] - 2*k[1,1]*np.tan(g[0])) + 4*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           -k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0])],
          [k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(3*l[1]) + k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           -k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) - k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           -k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(3*l[1]) + k[1,1]*np.tan(g[0])/2 + k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0])]])
