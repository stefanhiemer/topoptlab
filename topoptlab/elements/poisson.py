# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.fem import get_integrpoints
from topoptlab.elements.trilinear_hexahedron import invjacobian,shape_functions_dxi

def _lk_poisson(xe: np.ndarray, k: np.ndarray,
                quadr_method: str = "gauss-legendre",
                nquad: int = 2,
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for a Laplacian operator with linear elements.
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    k : np.ndarray, shape (nels,ndim,ndim) or
        conductivity tensor or something equivalent.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
        
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
    if len(k.shape) == 2:
        k = k[None,:,:]
    #
    x,w=get_integrpoints(ndim=ndim, nq=nquad, method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, 3,axis=1)]
    #
    Jinv,detJ = invjacobian(xi=xi, eta=eta, zeta=zeta,
                            xe=xe,
                            all_elems=True,
                            return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    gradN = shape_functions_dxi(xi=xi,eta=eta,zeta=zeta)[None,:,:,:]@\
            Jinv.transpose((0,1,3,2))
    #
    integral = gradN@k[:,None,:,:]@gradN.transpose([0,1,3,2])
    # multiply by determinant and quadrature
    return (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)