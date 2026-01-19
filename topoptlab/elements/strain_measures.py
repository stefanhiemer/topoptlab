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
    Return the the B matrix to calculate the infinitesimal or engineering 
    strain via from nodal displacements u:
        
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
    B : np.ndarray, shape (ncoords,(ndim**2 + ndim)/2,n_nodes*ndim) or 
        (nels,(ndim**2 + ndim)/2,n_nodes*ndim)
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
    
def disp_gradient(xi: np.ndarray, eta: np.ndarray, 
                  xe: np.ndarray,
                  invjacobian: np.ndarray, 
                  shape_functions_dxi: np.ndarray,
                  zeta: Union[None,np.ndarray] = None, 
                  check_fnc: Union[None,Callable] = None,
                  all_elems: bool = False, 
                  return_detJ: bool = False,
                  invJ: Union[None,np.ndarray] = None,
                  **kwargs: Any):
    """
    Return the the matrix B to calculate the flattened displacement gradient h
    ('C' ordering)from nodal displacements u:
        
        h = B_h@u
    
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
    invJ : None or np.ndarray
        inverse jacobian. If provided and return_detJ is False, uses this for 
        isoparametric mapping. Careful that the shapes are useable.
        
    Returns
    -------
    B_h : np.ndarray, shape (ncoords,ndim**2,n_nodes*ndim) or 
          (ndim**2,n_nodes*ndim)
        matrix for calculating the displacement gradient matrix.
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
    if not return_detJ and invJ is None:
        invJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                           return_det=return_detJ)
    elif return_detJ and invJ is not None:
        raise ValueError("The inputs do not make sense. disp_gradient computes",
                         "detJ as byproduct of inverting J. Please check your inputs. ")
    else:
        invJ,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                return_det=return_detJ)
    # helper array to collect shape function derivatives
    gradN = shape_functions_dxi(xi=xi,eta=eta)[None,:,:,:]@invJ.transpose((0,1,3,2))
    # empty def. grad. matrix
    B = np.zeros((invJ.shape[0], ndim**2, n_nodes*ndim))
    # tension components
    for i in np.arange(ndim): 
        for j in np.arange(ndim):
            B[:,i*ndim::(i+1)*ndim,i::ndim] = gradN
    if not return_detJ:
        return B
    else:
        return B, detJ
