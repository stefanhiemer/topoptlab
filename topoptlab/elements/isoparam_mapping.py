# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union
from warnings import warn

import numpy as np

def check_inputs(xi: Union[float,np.ndarray], eta: Union[float,np.ndarray], 
                 zeta: Union[None,np.ndarray] = None,
                 xe: Union[None,np.ndarray] = None,
                 all_elems: bool = False,
                 **kwargs: Any):
    """
    Check coordinates and provided element node information to be consistent.
    If necessary transform inputs to make them consistent.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be either 1,
        ncoords/n_nodes or the same as ncoords. The two exceptions are if
        ncoords = 1 or all_elems is True. Please look at the
        definition/function of the shape function, then the node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.

    Returns
    -------
    if xe is None
    ncoords : int
        number of coordinates
    if xe is not None
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (n).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (n).
    xe : np.ndarray
        coordinates of element nodes shape (n,4,2).
    None :
        for compatibility with 3D.
    """
    #
    if isinstance(xi,np.ndarray) and isinstance(eta,np.ndarray):
        #
        if len(xi.shape) != 1 or len(eta.shape) != 1:
            raise ValueError("xi and eta must be 1D: ",
                             xi.shape,eta.shape)
        elif xi.shape[0] != eta.shape[0]:
            raise ValueError("xi and eta must have same shape: ",
                             xi.shape,eta.shape)
        else:
            ncoords = xi.shape[0]
    elif (isinstance(xi,int) and isinstance(eta,int)) or\
         (isinstance(xi,float) and isinstance(eta,float)):
        ncoords = 1
    else:
        raise ValueError("Datatypes of xi and eta inconsistent.")
    if xe is not None:
        #
        xe_shape = xe.shape
        if len(xe_shape) == 2:
            xe = xe[None,:,:]
            xe_shape = xe.shape
        nels = xe.shape[0]
        if not all_elems and all([nels != ncoords,4*nels != ncoords,
                                  nels != 1,ncoords!=1]):
            raise ValueError("shapes of nels and ncoords incompatible.")
        elif all_elems:
            xi = np.tile(xi,nels)
            eta = np.tile(eta,nels)
            xe = np.repeat(xe,repeats=ncoords,axis=0)
        elif 4*nels == ncoords:
            xe = np.repeat(xe,repeats=4,axis=0)
        return xe,xi,eta,None
    else:
        return ncoords

def jacobian(xi,eta,xe,
             shape_functions_dxi,
             all_elems=False):
    """
    Jacobian for quadratic bilinear Lagrangian element.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,4,2). nels must be either 1,
        ncoords/4 or the same as ncoords. The two exceptions are if
        ncoords = 1 or all_elems is True.
        Please look at the definition/function of the shape function, then the
        node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.

    Returns
    -------
    J : np.ndarray, shape (ncoords,2,2) or (nels,2,2)
        Jacobian.

    """
    # check coordinates and node data for consistency
    xe,xi,eta,_ = check_inputs(xi=xi,eta=eta,xe=xe,all_elems=all_elems)
    return shape_functions_dxi(xi=xi,eta=eta).transpose([0,2,1]) @ xe

def invjacobian(xi,eta,xe,
                all_elems=False,return_det=False):
    """
    Inverse Jacobian for bilinear quadrilateral Lagrangian element.

    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,4,2). nels must be either 1,
        ncoords/4 or the same as ncoords. The two exceptions are if
        ncoords = 1 or all_elems is True.
        Please look at the definition/function of the shape function, then the
        node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.
    return_det : bool
        if True, return determinant of Jacobian.

    Returns
    -------
    Jinv : np.ndarray, shape (ncoords,2,2) or (nels,2,2)
           Jacobian.
    detJ : np.ndarray, shape (ncoords) or (nels)
           if return_det is True, determinant of Jacobian.

    """
    # jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=all_elems)
    # determinant
    detJ = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    # raise warning if determinant close to zero
    if np.any(np.isclose(detJ, 0)):
        warn("Determinant of element numerically close to zero.")
    elif np.any(detJ<0):
        raise ValueError("Determinant of Jacobian negative.")
    # adjungate matrix
    adj = np.zeros(J.shape)
    adj[:, 0, 0], adj[:, 1, 1] = J[:, 1, 1], J[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -J[:, 0, 1], -J[:, 1, 0]
    # return inverse
    if not return_det:
        return adj/detJ[:,None,None]
    else:
        return adj/detJ[:,None,None], detJ
