# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Union

import numpy as np

def eng_strain(xi: np.ndarray, eta: np.ndarray, 
               xe: np.ndarray,
               invjacobian: np.ndarray, 
               shape_functions_dxi: np.ndarray,
               zeta: Union[None,np.ndarray] = None, 
               check_fnc: Union[None,Callable] = None,
               all_elems: bool = False, return_detJ: bool = False,
               **kwargs: Any):
    """
    Engineering strain via the B matrix for bilinear quadrilateral Lagrangian 
    element to calculate to calculate strains, stresses etc. from nodal values:
    
    eps = B@u
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be 
        either 1, ncoords/4 or the same as ncoords. The two exceptions are if 
        ncoords = 1 or all_elems is True. 
        Please look at the definition/function of the shape function, then the 
        node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
        
    Returns
    -------
    B : np.ndarray, shape (ncoords,3,8) or (nels,3,8)
        B matrix.
    detJ : np.ndarray, shape (ncoords) or (nels)
           determinant of Jacobian.
        
    """
    #
    nel, n_nodes, ndim = xe.shape
    # check coordinates and node data for consistency
    if hasattr(check_fnc, '__call__') and zeta is None:
        xe,xi,eta,zeta = check_fnc(xi=xi,eta=eta,xe=xe,
                                   all_elems=all_elems) 
    # collect inverse jacobians
    if not return_detJ:
        invJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                           return_det=return_detJ)
    else:
        invJ,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                return_det=return_detJ)
    # helper array to collect shape function derivatives
    gradN = shape_functions_dxi(xi=xi,eta=eta)[None,:,:,:]@invJ.transpose((0,1,3,2))
    # empty small strain matrix
    B = np.zeros((invJ.shape[0], int((ndim**2 + ndim) /2), n_nodes*ndim))
    # tension components
    for i in np.arange(ndim): 
        B[:,i,i::ndim] = gradN[:,:,i]
    # shear components
    i,j = ndim-2,ndim-1
    for k in range(int((ndim**2 + ndim) /2) - ndim):
        #
        B[ndim+k][i::ndim] = gradN[:,:,j]
        B[ndim+k][j::ndim] = gradN[:,:,i]
        #
        i,j = (i+1)%ndim , (j+1)%ndim
    if not return_detJ:
        return B
    else:
        return B, detJ
