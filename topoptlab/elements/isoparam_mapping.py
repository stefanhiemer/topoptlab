# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Union
from warnings import warn

import numpy as np

def jacobian(xi: np.ndarray, 
             eta: np.ndarray, 
             xe: np.ndarray,
             shape_functions_dxi: Callable,
             zeta: Union[None,np.ndarray],
             all_elems: bool = False,
             **kwargs: Any) -> np.ndarray:
    """
    Jacobian for element.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be 
        either 1, ncoords/n_nodes or the same as ncoords. The two exceptions 
        are if ncoords = 1 or all_elems is True. 
    shape_functions_dxi : callable 
        gradient of shape functions with shape (ncoords,n_nodes,ndim)
    zeta : float or np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.

    Returns
    -------
    J : np.ndarray, shape (ncoords,ndim,ndim) or (nels,ndim,ndim)
        Jacobian.

    """
    # check coordinates and node data for consistency
    return shape_functions_dxi(xi=xi,eta=eta,zeta=zeta).transpose([0,2,1]) @ xe

def invjacobian(xe: np.ndarray,
                xi: Union[None,np.ndarray],
                eta: Union[None,np.ndarray],
                zeta: Union[None,np.ndarray],
                all_elems: bool=False,
                return_det: bool=False):
    """
    Inverse Jacobian for bilinear quadrilateral Lagrangian element.

    Parameters
    ----------
    xe : np.ndarray
        coordinates of element nodes shape (nels,nnodes,ndim). nels must be 
        either 1, ncoords/nnodes or the same as ncoords. The two exceptions are 
        if ncoords = 1 or all_elems is True.
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    zeta : np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.
    return_det : bool
        if True, return determinant of Jacobian.

    Returns
    -------
    Jinv : np.ndarray, shape (ncoords,ndim,ndim) or (nels,ndim,ndim)
           Jacobian.
    detJ : np.ndarray, shape (ncoords) or (nels)
           if return_det is True, determinant of Jacobian.

    """
    # jacobian
    J = jacobian(xi=xi,eta=eta,xe=xe,all_elems=all_elems)
    # determinant
    if eta is None:
        detJ = J[:,0,0]
    elif zeta is None:
        detJ = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    else:
        detJ = (J[:,0,0]*(J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1])-
                J[:,0,1]*(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])+
                J[:,0,2]*(J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0]))
    # raise warning if determinant close to zero
    if np.any(np.isclose(detJ, 0)):
        warn("Determinant of element numerically close to zero.")
    elif np.any(detJ<0):
        raise ValueError("Determinant of Jacobian negative.")
    # adjungate matrix
    adj = np.zeros(J.shape)
    if eta is None:
        adj[:]=1.
    elif zeta is None:
        adj[:, 0, 0], adj[:, 1, 1] = J[:, 1, 1], J[:, 0, 0]
        adj[:, 0, 1], adj[:, 1, 0] = -J[:, 0, 1], -J[:, 1, 0]
    else:
        adj[:,0,0] = J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1]
        adj[:,0,1] = -(J[:,0,1]*J[:,2,2] - J[:,0,2]*J[:,2,1])
        adj[:,0,2] = J[:,0,1]*J[:,1,2] - J[:,0,2]*J[:,1,1]

        adj[:,1,0] = -(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])
        adj[:,1,1] = J[:,0,0]*J[:,2,2] - J[:,0,2]*J[:,2,0]
        adj[:,1,2] = -(J[:,0,0]*J[:,1,2] - J[:,0,2]*J[:,1,0])

        adj[:,2,0] = J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0]
        adj[:,2,1] = -(J[:,0,0]*J[:,2,1] - J[:,0,1]*J[:,2,0])
        adj[:,2,2] = J[:,0,0]*J[:,1,1] - J[:,0,1]*J[:,1,0]
    # return inverse
    if not return_det:
        return adj/detJ[:,None,None]
    else:
        return adj/detJ[:,None,None], detJ
