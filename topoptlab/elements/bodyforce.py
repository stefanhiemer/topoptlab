# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.trilinear_hexahedron import shape_functions,jacobian
from topoptlab.fem import get_integrpoints

def _lf_bodyforce(xe: np.ndarray,
                  b: np.ndarray = np.array([0,-1.,0.]),
                  quadr_method: str = "gauss-legendre",
                  nquad: int = 1,
                  **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on trilinear hexahedral Lagrangian element (1st order)
    due to bodyforce (e. g. gravity) via numerical integration.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    b : np.ndarray of shape (nels,ndim) or (ndim)
        density of element
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points

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
    if (len(b.shape) == 1) or (b.shape[0] == 1):
        b = np.full((xe.shape[0],3), b)
    #
    x,w=  get_integrpoints(ndim=ndim, nq=nquad, method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, 3, axis=1)]
    # shape functions have shape (nq,8)
    N = np.kron(shape_functions(xi=xi, eta=eta, zeta=zeta)[:,:,None], 
                np.eye(ndim))
    #
    integral = N[None,:,:,:] @ b[:,None,None,:].transpose(0,1,3,2)
    # calculate determinant of jacobian
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=True)
    detJ = (J[:,0,0]*(J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1])-
            J[:,0,1]*(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])+
            J[:,0,2]*(J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0])).reshape(nel,nq)
    # multiply by determinant and quadrature
    fe = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #
    return fe