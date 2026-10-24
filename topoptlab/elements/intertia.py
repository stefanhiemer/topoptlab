# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any
from collections.abc import Callable

import numpy as np

from topoptlab.fem import get_integrpoints


def lk_intertia(xe: np.ndarray,
                ve: np.ndarray,
                quadr_method: str = "gauss-legendre",
                nquad: int = 2,
                shape_functions: None | Callable = None,
                shape_functions_dxi: None | Callable = None,
                invjacobian: None | Callable = None,
                **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Element tangent matrix and convective residual vector for the inertia
    (convective) term of the Navier–Stokes equations.

    For a Newton linearisation around a velocity field v_h, the convective
    term (u·∇)u contributes two parts to the tangent matrix:

        Oseen term        :  ∫ φ_k (v_h_l ∂u_k/∂x_l) dΩ
        Newton correction :  ∫ φ_k (u_l  ∂v_h_k/∂x_l) dΩ

    and the nonlinear residual vector at the current iterate is

        ∫ φ_k (v_h_l ∂v_h_k/∂x_l) dΩ

    Parameters
    ----------
    xe : np.ndarray, shape (nels, n_nodes, ndim)
        Coordinates of element nodes. Please look at the definition/function
        of the shape function; the node ordering follows from there.
    ve : np.ndarray, shape (nels, n_nodes*ndim) or (ndim,)
        Nodal velocity values at the linearisation point. A 1-D array of
        length ``ndim`` is broadcast as a uniform velocity over all nodes and
        elements.
    quadr_method : str or callable
        Name of quadrature method, or a callable returning quadrature
        coordinates and weights. Check ``get_integrpoints`` for available
        options.
    nquad : int
        Number of quadrature points per direction.
    shape_functions : callable or None
        Returns shape-function values of shape ``(nq, n_nodes)``. When None,
        the standard element for the given ``ndim`` is used automatically
        (bilinear quadrilateral for ndim=2, trilinear hexahedron for ndim=3).
    shape_functions_dxi : callable or None
        Returns parametric gradients of shape functions of shape
        ``(nq, n_nodes, ndim)``. When None the standard element is used.
    invjacobian : callable or None
        Returns the inverse Jacobian of the isoparametric mapping and its
        determinant. When None the standard element is used.

    Returns
    -------
    Ke : np.ndarray, shape (nels, n_nodes*ndim, n_nodes*ndim)
        Element tangent matrix (Oseen + Newton correction).
    fe : np.ndarray, shape (nels, n_nodes*ndim)
        Element convective residual vector.
    """
    if len(xe.shape) == 2:
        xe = xe[None, :, :]
    nel, n_nodes, ndim = xe.shape
    # uniform velocity broadcast
    if ve.ndim == 1 and ve.shape[0] == ndim:
        ve = np.tile(ve, n_nodes)[None, :]
    elif ve.ndim == 2 and ve.shape != (nel, ndim * n_nodes):
        raise ValueError(f"ve shape should be (nels, n_nodes*ndim): {(nel, ndim*n_nodes)}. "
                         f"Current shape: {ve.shape}")
    # shape functions
    if shape_functions is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import shape_functions
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import shape_functions
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    # parametric gradients of shape functions
    if shape_functions_dxi is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    # inverse jacobian for isoparametric mapping
    if invjacobian is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import invjacobian
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import invjacobian
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    #
    x, w = get_integrpoints(ndim=ndim, nq=nquad, method=quadr_method)
    nq = w.shape[0]
    xi, eta, zeta = [x[:, i] for i in range(ndim)] + [None] * (3 - ndim)
    # scalar shape functions: (nq, n_nodes)
    N = shape_functions(xi=xi, eta=eta, zeta=zeta)
    # inverse Jacobian and determinant
    Jinv, detJ = invjacobian(xi=xi, eta=eta, zeta=zeta,
                             xe=xe,
                             all_elems=True,
                             return_det=True)
    Jinv = Jinv.reshape(nel, nq, ndim, ndim)
    detJ = detJ.reshape(nel, nq)
    # physical gradients of shape functions: (nel, nq, n_nodes, ndim)
    gradN = shape_functions_dxi(xi=xi, eta=eta, zeta=zeta)[None, :, :, :] @ Jinv.transpose((0, 1, 3, 2))
    # nodal velocities reshaped to (nel, n_nodes, ndim)
    ve_mat = ve.reshape(nel, n_nodes, ndim)
    # velocity at quadrature points: (nel, nq, ndim)
    v_h = np.einsum('qi,eid->eqd', N, ve_mat)
    # gradient of v_h: grad_v[e,q,k,l] = dv_k/dx_l,  shape (nel, nq, ndim, ndim)
    grad_v = np.einsum('eqil,eik->eqkl', gradN, ve_mat)
    # Oseen scalar kernel: K_s[e,i,j] = ∫ N_i (v_h·∇N_j) dΩ
    v_dot_gradN = np.einsum('eqd,eqjd->eqj', v_h, gradN)
    K_scalar = np.einsum('q,eq,qi,eqj->eij', w, detJ, N, v_dot_gradN)
    # Oseen block-diagonal: K[e, i*ndim+k, j*ndim+l] = δ_{kl} K_s[e,i,j]
    Ke = np.einsum('eij,kl->eikjl', K_scalar, np.eye(ndim)).reshape(nel, n_nodes * ndim, n_nodes * ndim)
    # Newton correction: K[e, i*ndim+k, j*ndim+l] = ∫ N_i (∂v_k/∂x_l) N_j dΩ
    Ke += np.einsum('q,eq,eqkl,qi,qj->eikjl', w, detJ, grad_v, N, N).reshape(nel, n_nodes * ndim, n_nodes * ndim)
    # convective residual: fe[e, i*ndim+k] = ∫ N_i (v_h·∇v_h)_k dΩ
    conv = np.einsum('eql,eqkl->eqk', v_h, grad_v)
    fe = np.einsum('q,eq,qi,eqk->eik', w, detJ, N, conv).reshape(nel, n_nodes * ndim)
    return Ke, fe
