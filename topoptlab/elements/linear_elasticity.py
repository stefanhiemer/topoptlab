# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union
from warnings import warn

import numpy as np

from topoptlab.fem import get_integrpoints

def _lk_linear_elast(xe: np.ndarray,
                     c: np.ndarray,
                     quadr_method: str = "gauss-legendre",
                     nquad: int = 2,
                     bmatrix: Union[None,Callable] = None,
                     **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for linear elasticity with Lagrangian elements.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    c : np.ndarray
        stiffness tensor of shape (nels,(ndim**2 + ndim)/2,(ndim**2 + ndim)/2) or
        shape ((ndim**2 + ndim)/2,(ndim**2 + ndim)/2).
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points.
    bmatrix : callable or None
        function to calculate the B matrix (and determinant of the
        jacobian). If None, the function belonging to the reference element
        for the given ndim is imported automatically (bilinear_quadrilateral
        for ndim=2, trilinear_hexahedron for ndim=3).

    Returns
    -------
    Ke : np.ndarray, shape (nels,n_nodes*ndim,n_nodes*ndim)
        element stiffness matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel,n_nodes,ndim = xe.shape
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    # B matrix for isoparametric mapping
    if bmatrix is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import bmatrix
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import bmatrix
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [x[:,i] for i in range(ndim)] + [None]*(3-ndim)
    #
    B,detJ = bmatrix(xi=xi, eta=eta, zeta=zeta,
                     xe=xe,
                     all_elems=True,
                     return_detJ=True)
    detJ = detJ.reshape(nel,nq)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]@B
    # multiply by determinant and quadrature
    return (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)

def _lf_strain(xe: np.ndarray,
               eps: np.ndarray,
               c: np.ndarray,
               quadr_method: str = "gauss-legendre",
               nquad: int = 2,
               bmatrix: Union[None,Callable] = None,
               **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on Lagrangian elements due to a uniform strain
    via numerical integration for anisotropic linear elasticity.

    Parameters
    ----------
    xe : np.ndarray
        coordinates of element nodes of shape (nels,n_nodes,ndim).
        Please look at the definition/function of the shape function,
        then the node ordering is clear.
    eps : np.ndarray
        uniform strain in Voigt notation of shape (nels,(ndim**2 + ndim)/2) or ((ndim**2 + ndim)/2).
    c : np.ndarray
        stiffness tensor of shape (nels,(ndim**2 + ndim)/2,(ndim**2 + ndim)/2) or
        shape ((ndim**2 + ndim)/2,(ndim**2 + ndim)/2).
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
    bmatrix : callable or None
        function to calculate the B matrix (and determinant of the
        jacobian). If None, the function belonging to the reference element
        for the given ndim is imported automatically (bilinear_quadrilateral
        for ndim=2, trilinear_hexahedron for ndim=3).

    Returns
    -------
    fe : np.ndarray, shape (nels,n_nodes*ndim,1)
        nodal forces.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel,n_nodes,ndim = xe.shape
    #
    if (len(eps.shape) == 1) or (eps.shape[0] == 1):
        eps = np.full((xe.shape[0],int((ndim**2 + ndim)/2)), eps)
    #
    if len(c.shape) == 2:
        c = c[None,:,:]
    # B matrix for isoparametric mapping
    if bmatrix is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import bmatrix
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import bmatrix
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi, eta, zeta = [x[:,i] for i in range(ndim)] + [None]*(3-ndim)
    #
    B,detJ = bmatrix(xi=xi, eta=eta, zeta=zeta,
                     xe=xe,
                     all_elems=True,
                     return_detJ=True)
    detJ = detJ.reshape(nel,nq)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    #
    integral = B.transpose([0,1,3,2])@c[:,None,:,:]@eps[:,None,None,:].transpose(0,1,3,2)
    # multiply by determinant and quadrature
    return (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)