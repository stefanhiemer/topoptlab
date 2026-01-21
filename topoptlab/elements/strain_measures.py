# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Tuple, Union

import numpy as np

from topoptlab.elements.check_functions import check_inputs

def infini_strain_matrix(xi: np.ndarray, 
                         eta: Union[None,np.ndarray], 
                         zeta: Union[None,np.ndarray], 
                         xe: np.ndarray,
                         invjacobian: Union[Callable,np.ndarray], 
                         shape_functions_dxi: Union[Callable,np.ndarray],
                         all_elems: bool = False, 
                         return_detJ: bool = False,
                         check_fnc: Callable = check_inputs,
                         **kwargs: Any):
    """
    Return the the B matrix to calculate the infinitesimal or engineering 
    strain in Voigt notation via from nodal displacements u:
        
        eps = B@u
    
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
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping.
    shape_functions_dxi: callable or np.ndarray
        function to calculate the gradient of the shape functions.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
    check_fnc : callable
        function that checks for type and shape consistency of the inputs.
        
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
    xe,xi,eta,zeta = check_fnc(xi,eta,zeta,
                               ndim=ndim,
                               nnodes=n_nodes,
                               xe=xe,
                               all_elems=all_elems)
    # collect inverse jacobian
    invJ,detJ = _collect_invjacobian(xi=xi, 
                                     eta=eta, 
                                     xe=xe,
                                     invjacobian=invjacobian,  
                                     zeta=zeta, 
                                     return_detJ=return_detJ)
    # collect shape function derivatives and apply isoparametric map
    gradN=shape_functions_dxi(xi=xi,eta=eta,zeta=zeta)@invJ.transpose((0,2,1))
    # empty small strain matrix
    B = np.zeros((invJ.shape[0], int((ndim**2 + ndim) /2), n_nodes*ndim))
    # tension components
    for i in np.arange(ndim): 
        B[:,i,i::ndim] = gradN[:,:,i]
    # shear components
    i,j = ndim-2,ndim-1
    for k in range(int((ndim**2 + ndim) /2) - ndim):
        #
        B[:,ndim+k,i::ndim] = gradN[:,:,j]
        B[:,ndim+k,j::ndim] = gradN[:,:,i]
        #
        i,j = (i+1)%ndim , (j+1)%ndim
    if not return_detJ:
        return B
    else:
        return B, detJ
    
def dispgrad_matrix(xi: np.ndarray, 
                    eta: Union[None,np.ndarray], 
                    zeta: Union[None,np.ndarray], 
                    xe: np.ndarray,
                    invjacobian: Union[Callable,np.ndarray], 
                    shape_functions_dxi: Union[Callable,np.ndarray],
                    all_elems: bool = False, 
                    return_detJ: bool = False,
                    check_fnc: Callable = check_inputs,
                    **kwargs: Any):
    """
    Return the the matrix B to calculate the flattened displacement gradient h
    ('C' ordering)from nodal displacements u:
        
        h = B_h@u
    
    The matrix form can then be reconstructed by 
        
        H = h.reshape((-1,ndim,ndim),order="C")
    
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
    invjacobian : callable or np.ndarray
        Either function to calculate the inverse jacobian for the isoparametric 
        mapping or already calculated inverse. If the latter is the case, be 
        careful that the shape is consistent with the all_elems argument.
    shape_functions_dxi: callable or np.ndarray
        function to calculate the gradient of the shape functions.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
    check_fnc : callable
        function that checks for type and shape consistency of the inputs.
        
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
    xe,xi,eta,zeta = check_fnc(xi,
                               eta,
                               zeta,
                               ndim=ndim, 
                               nnodes=n_nodes,
                               xe=xe,
                               all_elems=all_elems)
    # collect inverse jacobian
    invJ,detJ = _collect_invjacobian(xi=xi, 
                                     eta=eta, 
                                     zeta=zeta,
                                     xe=xe,
                                     invjacobian=invjacobian,  
                                     return_detJ=return_detJ)
    # collect shape function derivatives and apply isoparametric map
    gradN=shape_functions_dxi(xi=xi,eta=eta,zeta=zeta)@invJ.transpose((0,2,1))
    # empty def. grad. matrix
    B = np.zeros((invJ.shape[0], ndim**2, n_nodes*ndim))
    # 
    for i in np.arange(ndim): 
        for j in np.arange(ndim):
            B[:,i*ndim + j,i::ndim] = gradN[:,:,j]
    if not return_detJ:
        return B
    else:
        return B, detJ

def lagrangian_strainvar_matrix(xi: np.ndarray, 
                                eta: Union[None,np.ndarray], 
                                zeta: Union[None,np.ndarray],
                                xe: np.ndarray,
                                F: np.ndarray,
                                invjacobian: Union[Callable,np.ndarray], 
                                shape_functions_dxi: Union[Callable,np.ndarray],
                                all_elems: bool = False, 
                                return_detJ: bool = False,
                                check_fnc: Callable = check_inputs,
                                **kwargs: Any):
    """
    Return the the B matrix to calculate the variation of the Lagrangian 
    strain E in Voigt notation from nodal displacements u:
            
        var(E) = B_dE@u
        
    The Lagrangian strain reads as: 
        
        E = 1/2 * ( C - I )
    
    I is the identity matrix and C the Cauchy–Green deformation tensor:
        
        C = F.T @ F
        
    where F is the deformation gradinet in matrix notation.
    
    
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
    F : None or np.ndarray
        deformation gradient of shape (nel,ndim,ndim)
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping.
    shape_functions_dxi: callable or np.ndarray
        function to calculate the gradient of the shape functions.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
    return_detJ : bool
        if True, return determinant of jacobian.
    check_fnc : callable
        function that checks for type and shape consistency of the inputs.
        
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
    xe,xi,eta,zeta = check_fnc(xi,eta,zeta,
                               ndim=ndim, 
                               nnodes=n_nodes,
                               xe=xe,
                               all_elems=all_elems)
    # collect inverse jacobian
    invJ,detJ = _collect_invjacobian(xi=xi, 
                                     eta=eta, 
                                     zeta=zeta, 
                                     xe=xe,
                                     invjacobian=invjacobian,  
                                     return_detJ=return_detJ) 
    # collect shape function derivatives and apply isoparametric map
    gradN=shape_functions_dxi(xi=xi,eta=eta,zeta=zeta)@invJ.transpose((0,2,1))
    # empty small strain matrix
    B = np.zeros((invJ.shape[0], int((ndim**2 + ndim) /2), n_nodes*ndim))
    # tension components
    for i in np.arange(ndim): 
        for j in np.arange(ndim):
            B[:,j,i::ndim] = F[:,None,i,j]*gradN[:,:,j]
    # shear components
    if ndim==2:
        inds=[[0],[1]]
    elif ndim==3:
        inds=[[1,0,0],[2,2,1]]
    for k in range(int((ndim**2 + ndim) /2) - ndim):
        for l in range(int((ndim**2 + ndim) /2) - ndim):
            #
            B[:,ndim+l,k::ndim] = F[:,None,k,inds[0][l]]*gradN[:,:,inds[1][l]]+\
                                  F[:,None,k,inds[1][l]]*gradN[:,:,inds[0][l]]
    if not return_detJ:
        return B
    else:
        return B, detJ

def _collect_invjacobian(xi: np.ndarray, 
                         eta: Union[None,np.ndarray], 
                         zeta: Union[None,np.ndarray], 
                         xe: np.ndarray,
                         invjacobian: Union[Callable,np.ndarray],  
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
    invjacobian : callable or np.ndarray
        function to calculate the inverse jacobian for the isoparametric 
        mapping. 
    return_detJ : bool
        if True, return determinant of jacobian. 
        
    Returns
    -------
    B : np.ndarray, shape (ncoords,(ndim**2 + ndim)/2,n_nodes*ndim) or 
        (nels,(ndim**2 + ndim)/2,n_nodes*ndim)
        B matrix.
    detJ : None or np.ndarray, shape (ncoords) or (nels)
           determinant of Jacobian. If return_detJ is False, then detJ is None.
        
    """
    #
    if not return_detJ:
        detJ = None
    #
    if not return_detJ and hasattr(invjacobian, '__call__'):
        invJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                           return_det=return_detJ)
    elif return_detJ and hasattr(invjacobian, '__call__'):
        invJ,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                return_det=return_detJ)
    elif return_detJ and isinstance(invjacobian, np.ndarray):
        raise ValueError("The inputs do not make sense. disp_gradient computes",
                         "detJ as byproduct of inverting J and you have already", 
                         " inverted J. Please also supply detJ. ")
    elif not return_detJ and isinstance(invjacobian, np.ndarray):
        invJ = invjacobian
    return invJ, detJ

if __name__ == "__main__":
    
    from topoptlab.elements.bilinear_quadrilateral import invjacobian,\
                                                          shape_functions_dxi,\
                                                          bmatrix
    xe = np.array([[[-2.1,-1],[1,-1],[1,1],[-1,1]]])
    bmat = infini_strain_matrix(eta=np.array([1.]),
                                xi=np.array([1.]),
                                zeta=None,
                                xe=xe, all_elems=False,
               invjacobian=invjacobian,
               shape_functions_dxi=shape_functions_dxi,
               check_fnc=check_inputs)
    hmat = dispgrad_matrix(eta=1.,xi=1.,zeta=None,
                           xe=xe, all_elems=False,
                           invjacobian=invjacobian,
                           shape_functions_dxi=shape_functions_dxi)
    smat = lagrangian_strainvar_matrix(eta=1.,xi=1.,zeta=None,
                                       F=np.eye(2)[None,:,:],
                                       xe=xe, all_elems=False,
               invjacobian=invjacobian,
               shape_functions_dxi=shape_functions_dxi)
    print(bmat.shape)
    print(hmat.shape)
    print(smat.shape)
    ue=np.array([-2.,0.,-2.,0.,2.,0.,2.,0.])[None,:]
    print(bmat@ue.T)
    print(hmat@ue.T)
    print(smat@ue.T)