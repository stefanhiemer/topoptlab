# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any
from collections.abc import Callable

import numpy as np

from topoptlab.fem import get_integrpoints

def _lf_bodyforce(xe: np.ndarray,
                  b: None | np.ndarray = None,
                  quadr_method: str = "gauss-legendre",
                  nquad: int = 1,
                  shape_functions: None | Callable = None,
                  jacobian: None | Callable = None,
                  **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on Lagrangian elements (1st order) due to
    bodyforce (e. g. gravity) via numerical integration.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    b : np.ndarray of shape (nels,ndim) or (ndim) or None
        density of element. If None, defaults to unit gravity in -y direction
        (0,-1) or (0,-1,0), depending on ndim.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
    shape_functions : callable or None
        function to calculate the shape functions. If None, the function
        belonging to the reference element for the given ndim is imported
        automatically (bilinear_quadrilateral for ndim=2, trilinear_hexahedron
        for ndim=3).
    jacobian : callable or None
        function to calculate the jacobian of the isoparametric mapping. If
        None, the function belonging to the reference element for the given
        ndim is imported automatically (bilinear_quadrilateral for ndim=2,
        trilinear_hexahedron for ndim=3).

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
    if b is None:
        b = np.array([0.,-1.] + [0.]*(ndim-2))
    if (len(b.shape) == 1) or (b.shape[0] == 1):
        b = np.full((xe.shape[0],ndim), b)
    # shape functions
    if shape_functions is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import shape_functions
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import shape_functions
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    # jacobian of isoparametric mapping
    if jacobian is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import jacobian
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import jacobian
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
    #
    x,w=  get_integrpoints(ndim=ndim, nq=nquad, method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [x[:,i] for i in range(ndim)] + [None]*(3-ndim)
    # shape functions have shape (nq,n_nodes)
    N = np.kron(shape_functions(xi=xi, eta=eta, zeta=zeta)[:,:,None],
                np.eye(ndim))
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=True)
    detJ = np.linalg.det(J).reshape(nel,nq)
    #
    integral = N[None,:,:,:] @ b[:,None,None,:].transpose(0,1,3,2)
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return fe