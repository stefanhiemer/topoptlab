# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union

import numpy as np

from topoptlab.fem import get_integrpoints

def _lk_advection(xe: np.ndarray,
                 ve: np.ndarray,
                 quadr_method: str = "gauss-legendre",
                 nquad: int = 2,
                 shape_functions: Union[None,Callable] = None,
                 shape_functions_dxi: Union[None,Callable] = None,
                 invjacobian: Union[None,Callable] = None,
                 **kwargs: Any) -> np.ndarray:
    """
    Create element matrix for an advection operator given velocity v 
    and a scalar field T (temperature, concentration, etc.).

       v.T grad T.
    

    
    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ve : np.ndarray, shape (nels,n_nodes*ndim)
        nodal velocities.
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
    # uniform velocity
    if len(ve.shape) == 1 and ve.shape[0] == ndim:
        ve = np.tile(ve, n_nodes)[None,:]
    elif len(ve.shape) == 2 and ve.shape != (nel,ndim*n_nodes):
        raise ValueError(f"ve shape should be (nels,n_nodes*ndim): {(nel,ndim*n_nodes)}. "
                         f"Current shape: {ve.shape}")
    # shape functions
    if shape_functions is None:
        if ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import shape_functions
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import shape_functions
        else:
            raise NotImplementedError(f"only dimensions 2 and 3 implemented. current ndim {ndim}")
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
    # shape functions for velocities have shape (nq,n_nodes*ndim)
    N = shape_functions(xi=xi, eta=eta, zeta=zeta)[:,:,None]
    N_v = np.kron(N, np.eye(ndim))
    #
    Jinv,detJ = invjacobian(xi=xi, eta=eta, zeta=zeta,
                            xe=xe,
                            all_elems=True,
                            return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    gradN = shape_functions_dxi(xi=xi,eta=eta,zeta=zeta)[None,:,:,:]@\
            Jinv.transpose((0,1,3,2))
    # ue[:,None,:,None]
    integral = N[None,:,:,:]@ve[:,None,:,None].transpose([0,1,3,2])@N_v[None,:,:,:]\
               @gradN.transpose([0,1,3,2])
    # multiply by determinant and quadrature
    return (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)