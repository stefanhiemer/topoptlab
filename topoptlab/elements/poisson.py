# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union

import numpy as np

from topoptlab.fem import get_integrpoints

def _lk_poisson(xe: np.ndarray,
                k: np.ndarray,
                quadr_method: str = "gauss-legendre",
                nquad: int = 2,
                shape_functions_dxi: Union[None,Callable] = None,
                invjacobian: Union[None,Callable] = None,
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for a Laplacian operator with linear elements.
    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    k : np.ndarray, shape (nels,ndim,ndim) or (ndim,ndim)
        conductivity tensor or something equivalent.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
    shape_functions_dxi : callable or None
        function to calculate the gradient of the shape functions. If None,
        bilinear quadrilateral or trilinear hexahedral elements are assumed (ndim=2/3).
    invjacobian : callable or None
        function to calculate the inverse jacobian for the isoparametric
        mapping. If None, bilinear quadrilateral or trilinear hexahedral 
        elements are assumed (ndim=2/3).

    Returns
    -------
    Ke : np.ndarray, shape (nels,n_nodes*ndim,n_nodes*ndim)
        element stiffness matrix.
        
    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel, n_nodes, ndim = xe.shape
    #
    if len(k.shape) == 2:
        k = k[None,:,:]
    # gradient of shape functions
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
    x,w=get_integrpoints(ndim=ndim, nq=nquad, method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [x[:,i] for i in range(ndim)] + [None]*(3-ndim)
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