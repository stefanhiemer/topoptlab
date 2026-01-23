# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union

import numpy as np

from topoptlab.elements.isoparam_mapping import invjacobian, \
                                                _collect_invjacobian
from topoptlab.elements.check_functions import check_inputs

def hessian_matrix(xi: np.ndarray, 
                   eta: Union[None,np.ndarray], 
                   zeta: Union[None,np.ndarray],
                   xe: np.ndarray,
                   shape_functions_dxi: Union[Callable,np.ndarray],
                   shape_functions_hessian: Callable,
                   invjacobian: Union[Callable,np.ndarray] = invjacobian, 
                   all_elems: bool = False, 
                   return_detJ: bool = False,
                   check_fnc: Callable = check_inputs,
                   **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D HuHu regularization with
    bilinear quadrilateral Lagrangian elements:
        
        hessian = B_hessian@u_e

    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain.
    eta : None or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain. 
    zeta : None or np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates 
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be 
        either 1, ncoords/4 or the same as ncoords. The two exceptions are if 
        ncoords = 1 or all_elems is True. 
        Please look at the definition/function of the shape function, then the 
        node ordering is clear.
    shape_functions_dxi: callable
        function to calculate the gradient of the shape functions.
    shape_functions_hessian: callable
        function to calculate hessian of shape functions per shape 
        function/node at specified coordinate(s).
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
    check_fnc : callable
        function that checks for type and shape consistency of the inputs.
        
    Returns
    -------
    B_hessian : np.ndarray, shape (nels,ndim**3,nnodes*ndim)
        element stiffness matrix.

    """
    #
    nel, n_nodes, ndim = xe.shape
    # check coordinates and node data for consistency
    xe,xi,eta,zeta = check_fnc(xi,
                               eta,
                               zeta,
                               ndim=ndim, 
                               nnodes=n_nodes,
                               xe=xe,
                               all_elems=all_elems)
    # collect inverse jacobian
    JinvJ,Jdet = _collect_invjacobian(xi=xi, 
                                     eta=eta, 
                                     xe=xe,
                                     shape_functions_dxi=shape_functions_dxi,
                                     invjacobian=invjacobian,  
                                     zeta=zeta, 
                                     return_detJ=return_detJ)
    # collect hessian in ref. space
    B_hessian = shape_functions_hessian(xi=xi, eta=eta, zeta=zeta) 
    # apply isop. map
    B_hessian = Jdet.swapaxes(-1,-2)[:,:,None,:,:]@B_hessian[None,:,:,:]@\
                Jdet[:,:,None,:,:]
    # flatten hessian
    B_hessian = B_hessian.reshape(B_hessian.shape[:3]+tuple([ndim**2]))
    B_hessian = B_hessian.swapaxes(-1,-2)
    B_hessian = np.kron(B_hessian,np.eye(ndim))
    return B_hessian