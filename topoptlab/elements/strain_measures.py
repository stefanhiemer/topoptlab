from warnings import warn

import numpy as np

def jacobian(shape_functions_dxi, 
             xi,eta,xe,all_elems=False,
             zeta=None, check_fnc=None):
    """
    Jacobian for parametric mapping. 
    
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
    xe,xi,eta,_ = check_fnc(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=all_elems) 
    return shape_functions_dxi(xi,eta).transpose([0,2,1]) @ xe 

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
    detJ = np.linalg.det(J)
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

def bmatrix(xi,eta,xe,
            invjacobian,shape_functions_dxi,
            zeta=None, check_fnc=None,
            all_elems=False, return_detJ=False):
    """
    B matrix for bilinear quadrilateral Lagrangian element to calculate
    to calculate strains, stresses etc. from nodal values
    
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
        xe,xi,eta,zeta = check_fnc(xi=xi,eta=eta,xe=xe,all_elems=all_elems) 
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
