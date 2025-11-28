# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple,Any,Union,List
from warnings import warn

import numpy as np

def create_edofMat(nelx: int, nely: int, nnode_dof: int,
                   dtype: type = np.int32, **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,None,None]:
    """
    Create element degree of freedom matrix for bilinear Lagrangian elements in
    a regular mesh.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nnode_dof : int
        number of node degrees of freedom.

    Returns
    -------
    edofMat : np.ndarray
        element degree of freedom matrix
    n1 : np.ndarray
        index array to help constructing the stiffness matrix.
    n2 : np.ndarray
        index array to help constructing the stiffness matrix.
    n3 :  None
        purely there for compatibility with 3 dimensions
    n4 :  None
        purely there for compatibility with 3 dimensions
    """
    # create arrays for indexing
    elx = np.arange(nelx, dtype=dtype)[:,None]
    ely = np.arange(nely, dtype=dtype)[None,:]
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    #
    edofMat = np.column_stack((n1+1,n2+1,n2,n1))*nnode_dof
    edofMat = np.repeat(edofMat,nnode_dof,axis=1)
    edofMat = edofMat + np.tile(np.arange(nnode_dof,dtype=dtype),4)[None,:]
    return edofMat, n1, n2, None, None

def apply_pbc(edofMat: np.ndarray, pbc: Union[List,np.ndarray],
              nelx: int, nely: int, nnode_dof: int,
              dtype: type = np.int32, **kwargs: Any
              ) -> np.ndarray:
    """
    Convert a given element-degree-of-freedom matrix (edofMat) of a regular
    mesh of first order Lagrangian quadrilateral elements with free
    boundary conditions to the corresponding edofMat with periodic boundary
    conditions by re-labelling the nodal degrees of freedom

    Parameters
    ----------
    edofMat : np.ndarray of shape (nel,4*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular
        mesh of first order Lagrangian quadrilateral elements with free
        boundary conditions.
    pbc : list or np.ndarray of shape/len 2 with elements of datatype bool
        periodic boundary condition flags.
    nelx : int
        number of elements in x-direction.
    nely : int
        number of elements in y-direction.
    nnode_dof : int
        number of degrees of freedom per node.
    dtype : type
        datatype of element degrees of freedom. Should be just large enough to
        store the highest number of the degrees of freedom to save memory. For
        practical purposes np.int32 should do the job.

    Returns
    -------
    edofMat_new : np.ndarray of shape (nel,4*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular
        mesh of first order Lagrangian quadrilateral elements with periodic
        boundary conditions with re-labelled nodal degrees of freedom.
    """
    # update indices
    if pbc[1]:
        edofMat -= np.floor(edofMat / (nnode_dof*(nely+1)) \
                            ).astype(dtype)*nnode_dof
    #
    nel = nelx*nely
    # x
    if pbc[0]:
        # reassign indices
        org = np.arange(nely)
        pbc_x = np.arange(nel-nely,nel)
        edofMat[pbc_x,nnode_dof:2*nnode_dof] = edofMat[org,:nnode_dof]
        edofMat[pbc_x,2*nnode_dof:3*nnode_dof] = edofMat[org,-nnode_dof:]
    # y
    if pbc[1]:
        # reassign indices
        org = np.arange(0,nelx*nely,nely)
        pbc_y = np.arange(nely-1,nelx*nely+1,nely)
        edofMat[pbc_y,:nnode_dof] = edofMat[org,-nnode_dof:]
        edofMat[pbc_y,nnode_dof:2*nnode_dof] = edofMat[org,2*nnode_dof:3*nnode_dof]
    return edofMat

def check_inputs(xi: Union[float,np.ndarray],
                 eta: Union[float,np.ndarray],
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
    xe : np.ndarray
        coordinates of element nodes shape (nels,4,2). nels must be either 1,
        ncoords/4 or the same as ncoords. The two exceptions are if
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
    #
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

def shape_functions(xi: np.ndarray, eta: np.ndarray,
                    **kwargs: Any) -> np.ndarray:
    """
    Shape functions for bilinear quadrilateral Lagrangian element in reference
    domain. Coordinates bounded in [-1,1].

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.

    Returns
    -------
    shape_functions : np.ndarray, shape (ncoords,4)
        values of shape functions at specified coordinate(s).

    """
    return 1/4 * np.column_stack(((1-xi)*(1-eta),
                                  (1+xi)*(1-eta),
                                  (1+xi)*(1+eta),
                                  (1-xi)*(1+eta)))

def shape_functions_dxi(xi: np.ndarray,eta: np.ndarray,
                        **kwargs: Any) -> np.ndarray:
    """
    Gradient of shape functions for bilinear quadrilateral Lagrangian element.
    The derivative is taken with regards to the reference coordinates, not the
    physical coordinates.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates
        are assumed to be in the reference domain.

    Returns
    -------
    shape_functions_dxi : np.ndarray, shape (ncoords,4,2)
        gradient of shape functions at specified coordinate(s).

    """
    dx = 1/4 * np.column_stack((eta-1, (xi-1),
                                1-eta, -1-xi,
                                1+eta, 1+xi,
                                -1-eta, 1-xi))
    return dx.reshape(int(np.prod(dx.shape)/8),4,2)

def jacobian(xi: np.ndarray, eta: np.ndarray, xe: np.ndarray,
             all_elems: bool = False) -> np.ndarray:
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

def invjacobian(xi: np.ndarray, eta: np.ndarray, xe: np.ndarray,
                all_elems: bool = False, return_det: bool = False
                ) -> np.ndarray:
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

def jacobian_rectangle(a: float, b: float) -> np.ndarray:
    """
    Jacobian for rectangular quadratic bilinear Lagrangian element.

    Parameters
    ----------
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.

    Returns
    -------
    J : np.ndarray, shape (2,2)
        Jacobian.

    """
    return 1/2 * np.array([[a,0],[0,b]])

def invjacobian_rectangle(a: float, b: float) -> np.ndarray:
    """
    Inverse Jacobian for rectangular quadratic bilinear Lagrangian element.

    Parameters
    ----------
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.

    Returns
    -------
    J : np.ndarray, shape (2,2)
        Jacobian.

    """
    return 2 * np.array([[1/a,0],[0,1/b]])

def bmatrix(xi: np.ndarray, eta: np.ndarray, xe: np.ndarray,
            all_elems: bool = False, return_detJ: bool = False
            ) -> np.ndarray:
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
        coordinates of element nodes shape (nels,4,2). nels must be either 1,
        ncoords/4 or the same as ncoords. The two exceptions are if
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
        infinitesimal strain matrix or B matrix.
    detJ : np.ndarray, shape (ncoords) or (nels)
           determinant of Jacobian.

    """
    # check coordinates and node data for consistency
    xe,xi,eta,_ = check_inputs(xi=xi,eta=eta,xe=xe,all_elems=all_elems)
    # collect inverse jacobians
    if not return_detJ:
        invJ = invjacobian(xi=xi,eta=eta,xe=xe,
                           return_det=return_detJ)
    else:
        invJ,detJ = invjacobian(xi=xi,eta=eta,xe=xe,
                                return_det=return_detJ)
    # helper array to collect shape function derivatives
    helper = np.zeros((invJ.shape[0],4,8))
    shp = shape_functions_dxi(xi=xi,eta=eta).transpose([0,2,1])
    helper[:,:2,::2] = shp
    helper[:,2:,1::2] = shp.copy() # copy to avoid np.views
    #
    B = np.array([[1,0,0,0],
                  [0,0,0,1],
                  [0,1,1,0]])@np.kron(np.eye(2),invJ)@helper
    if not return_detJ:
        return B
    else:
        return B, detJ

def bmatrix_rectangle(xi: np.ndarray, eta: np.ndarray,
                      a: float, b: float) -> np.ndarray:
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
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.

    Returns
    -------
    B : np.ndarray, shape (ncoords,3,8)
        B matrix.

    """
    # check coordinates for consistency
    ncoords = check_inputs(xi=xi,eta=eta)
    # collect inverse jacobians
    invJ = invjacobian_rectangle(a=a,b=b)
    # helper array to collect shape function derivatives
    helper = np.zeros((ncoords,4,8))
    shp = shape_functions_dxi(xi=xi,eta=eta).transpose([0,2,1])
    helper[:,:2,::2] = shp
    helper[:,2:,1::2] = shp.copy() # copy to avoid np.views
    #
    B = np.array([[1,0,0,0],
                  [0,0,0,1],
                  [0,1,1,0]])@np.kron(np.eye(2),invJ)@helper
    return B
