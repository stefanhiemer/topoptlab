# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Tuple,Union
from warnings import warn

import numpy as np

def jacobian(xi: Union[float,np.ndarray], 
             eta: Union[None,float,np.ndarray], 
             zeta: Union[None,float,np.ndarray],
             xe: np.ndarray,
             shape_functions_dxi: Callable,
             **kwargs: Any) -> np.ndarray:
    """
    Jacobian for element.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : None or float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    zeta : None or float or np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,n_nodes,ndim). nels must be 
        either 1, ncoords/n_nodes or the same as ncoords. The two exceptions 
        are if ncoords = 1 or all_elems is True. 
    shape_functions_dxi : callable 
        gradient of shape functions with shape (ncoords,n_nodes,ndim)

    Returns
    -------
    J : np.ndarray, shape (ncoords,ndim,ndim) or (nels,ndim,ndim)
        Jacobian.

    """
    #
    return shape_functions_dxi(xi=xi,
                               eta=eta,
                               zeta=zeta).transpose([0,2,1])@xe

def invjacobian(xi: Union[float,np.ndarray], 
                eta: Union[None,float,np.ndarray], 
                zeta: Union[None,float,np.ndarray], 
                xe: np.ndarray,
                shape_functions_dxi: Callable,
                all_elems: bool=False,
                return_det: bool=False):
    """
    Inverse Jacobian for bilinear quadrilateral Lagrangian element.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : None or float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    zeta : None or float or np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,nnodes,ndim). nels must be 
        either 1, ncoords/nnodes or the same as ncoords. The two exceptions are 
        if ncoords = 1 or all_elems is True.
    shape_functions_dxi : callable 
        gradient of shape functions with shape (ncoords,n_nodes,ndim)
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
    J = jacobian(xi=xi,eta=eta,zeta=zeta,
                 xe=xe,
                 shape_functions_dxi=shape_functions_dxi)
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

def _collect_invjacobian(xi: np.ndarray, 
                         eta: Union[None,np.ndarray], 
                         zeta: Union[None,np.ndarray], 
                         xe: np.ndarray,
                         shape_functions_dxi: Callable,
                         invjacobian: Union[Callable,np.ndarray] = invjacobian,  
                         return_detJ: bool = False,
                         **kwargs: Any) -> Tuple[np.ndarray, 
                                                Union[None,np.ndarray]]:
    """
    Internal helper function to collect the inverse jacobian to construct 
    different strain measures.
    
    
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
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping. 
    return_detJ : bool
        if True, return determinant of jacobian. 
        
    Returns
    -------
    invJ : np.ndarray, 
        inverse jacobian shape (ncoords,(ndim**2 + ndim)/2,n_nodes*ndim) or 
        (nels,ndim,ndim).
    detJ : None or np.ndarray
           determinant of Jacobian. If return_detJ is False, then detJ is None.
           shape (ncoords) or (nels)
        
    """
    #
    if not return_detJ:
        detJ = None
    #
    if not return_detJ and hasattr(invjacobian, '__call__'):
        invJ = invjacobian(xi=xi,eta=eta,zeta=zeta,
                           xe=xe,
                           shape_functions_dxi=shape_functions_dxi,
                           return_det=return_detJ)
    elif return_detJ and hasattr(invjacobian, '__call__'):
        invJ,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,
                                xe=xe,
                                shape_functions_dxi=shape_functions_dxi,
                                return_det=return_detJ)
    elif return_detJ and isinstance(invjacobian, np.ndarray):
        raise ValueError("The inputs do not make sense. disp_gradient computes",
                         "detJ as byproduct of inverting J and you have already", 
                         " inverted J. Please also supply detJ. ")
    elif not return_detJ and isinstance(invjacobian, np.ndarray):
        invJ = invjacobian
    return invJ, detJ