# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union

import numpy as np

from topoptlab.fem import get_integrpoints

def _lm_mass(xe: np.ndarray,
            p: np.ndarray = np.array([1.]),
            quadr_method: str = "gauss-legendre",
            nquad: int = 2,
            shape_functions: Union[None,Callable] = None,
            jacobian: Union[None,Callable] = None,
            **kwargs: Any) -> np.ndarray:
    """
    Create element mass matrix for scalar field with Lagrangian elements.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
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
    Ke : np.ndarray, shape (nels,n_nodes,n_nodes)
        element mass matrix.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel,n_nodes,ndim = xe.shape
    #
    if isinstance(p,float) or (p.shape[0] == 1 and xe.shape[0] !=1):
        p = np.full(xe.shape[0], p)
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
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [x[:,i] for i in range(ndim)] + [None]*(3-ndim)
    #
    N = shape_functions(xi=xi,eta=eta,zeta=zeta)
    #
    integral = N[None,:,:,None]@N[None,:,:,None].transpose([0,1,3,2])
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=True)
    detJ = np.linalg.det(J).reshape(nel,nq)
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return p[:,None,None] * Ke
