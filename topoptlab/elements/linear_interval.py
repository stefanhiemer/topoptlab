# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple,Any,Union,List
from warnings import warn

import numpy as np

def create_edofMat(nelx: int, 
                   nnode_dof: int,
                   dtype: type = np.int32, 
                   **kwargs: Any
                   ) -> Tuple[np.ndarray,None,None,None,None]:
    """
    Create element degree of freedom matrix for linear Lagrangian elements in
    a regular mesh.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nnode_dof : int
        number of node degrees of freedom.

    Returns
    -------
    edofMat : np.ndarray
        element degree of freedom matrix
    n1 : None
        purely there for compatibility with 2/3 dimensions.
    n2 : None
        purely there for compatibility with 2/3 dimensions.
    n3 :  None
        purely there for compatibility with 3 dimensions.
    n4 :  None
        purely there for compatibility with 3 dimensions.
    """
    # create arrays for indexing
    elx = np.arange(nelx, dtype=dtype)[:,None]
    #
    edofMat = np.column_stack((elx,elx+1))*nnode_dof
    edofMat = np.repeat(edofMat,nnode_dof,axis=1)
    edofMat = edofMat + np.tile(np.arange(nnode_dof,dtype=dtype),2)[None,:]
    return edofMat, None, None, None, None

def apply_pbc(edofMat: np.ndarray, 
              pbc: Union[List,np.ndarray],
              nelx: int, 
              nnode_dof: int,
              dtype: type = np.int32, 
              **kwargs: Any) -> np.ndarray:
    """
    Convert a given element-degree-of-freedom matrix (edofMat) of a regular
    mesh of first order Lagrangian interval with free boundary conditions to 
    the corresponding edofMat with periodic boundary conditions by re-labelling 
    the nodal degrees of freedom

    Parameters
    ----------
    edofMat : np.ndarray of shape (nel,2*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular
        mesh of first order Lagrangian quadrilateral elements with free
        boundary conditions.
    pbc : list or np.ndarray of shape/len 1 with elements of datatype bool
        periodic boundary condition flags.
    nelx : int
        number of elements in x-direction.
    nnode_dof : int
        number of degrees of freedom per node.
    dtype : type
        datatype of element degrees of freedom. Should be just large enough to
        store the highest number of the degrees of freedom to save memory. For
        practical purposes np.int32 should do the job.

    Returns
    -------
    edofMat_new : np.ndarray of shape (nel,2*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular
        mesh of first order Lagrangian interval elements with periodic
        boundary conditions with re-labelled nodal degrees of freedom.
    """
    if pbc[0]:
        # reassign indices
        edofMat[-1,-nnode_dof] = edofMat[0,:nnode_dof]
    return edofMat

def check_inputs(xi: Union[float,np.ndarray],
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
    xe : np.ndarray
        coordinates of element nodes shape (nels,2,1). nels must be either 1,
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
    xe : np.ndarray
        coordinates of element nodes shape (n,2,2).
    None :
        for compatibility with 3D.
    """
    #
    if isinstance(xi,np.ndarray):
        #
        if len(xi.shape) != 1:
            raise ValueError("xi must be 1D: ",
                             xi.shape)
        else:
            ncoords = xi.shape[0]
    elif isinstance(xi,int) or isinstance(xi,float):
        ncoords = 1
    else:
        raise ValueError("Datatype of xi nconsistent.")
    #
    if xe is not None:
        #
        xe_shape = xe.shape
        if len(xe_shape) == 2:
            xe = xe[None,:,:]
            xe_shape = xe.shape
        nels = xe.shape[0]
        if not all_elems and all([nels != ncoords,2*nels != ncoords,
                                  nels != 1,ncoords!=1]):
            raise ValueError("shapes of nels and ncoords incompatible.")
        elif all_elems:
            xi = np.tile(xi,nels)
            xe = np.repeat(xe,repeats=ncoords,axis=0)
        elif 2*nels == ncoords:
            xe = np.repeat(xe,repeats=2,axis=0)
        return xe,xi,None,None
    else:
        return ncoords

def shape_functions(xi: Union[float,np.ndarray], 
                    **kwargs: Any) -> np.ndarray:
    """
    Shape functions for bilinear quadrilateral Lagrangian element in reference
    domain. Coordinates bounded in [-1,1].

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).

    Returns
    -------
    shape_functions : np.ndarray, shape (ncoords,2)
        values of shape functions at specified coordinate(s).

    """
    return 1/2 * np.column_stack(((1-xi),
                                  (1+xi)))

def shape_functions_dxi(xi: Union[float,np.ndarray],
                        **kwargs: Any) -> np.ndarray:
    """
    Gradient of shape functions for linear interval Lagrangian element.
    The derivative is taken with regards to the reference coordinates, not the
    physical coordinates.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).

    Returns
    -------
    shape_functions_dxi : np.ndarray, shape (ncoords,2,1)
        gradient of shape functions at specified coordinate(s).

    """
    if isinstance(xi,float):
        ncoords = 1
    elif isinstance(xi,np.ndarray):
        ncoords = xi.shape
    return 1/2 * np.array([[-1, 1]]) * np.ones(ncoords)[:,None,None]

def jacobian(xi: Union[float,np.ndarray], 
             xe: np.ndarray,
             all_elems: bool = False) -> np.ndarray:
    """
    Jacobian for linear interval Lagrangian element.

    Parameters
    ----------
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    xe : np.ndarray
        coordinates of element nodes shape (nels,2,1). nels must be either 1,
        ncoords/4 or the same as ncoords. The two exceptions are if
        ncoords = 1 or all_elems is True.
        Please look at the definition/function of the shape function, then the
        node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for
        creating elements etc.

    Returns
    -------
    J : np.ndarray, shape (ncoords,1,1) or (nels,1,1)
        Jacobian.

    """
    # check coordinates and node data for consistency
    xe,xi,_,_ = check_inputs(xi=xi,xe=xe,all_elems=all_elems)
    print(xe.shape,shape_functions_dxi(xi=xi).shape)
    return shape_functions_dxi(xi=xi).transpose([0,2,1]) @ xe

def invjacobian(xi: np.ndarray, 
                xe: np.ndarray,
                all_elems: bool = False, 
                return_det: bool = False,
                **kwargs: Any) -> np.ndarray:
    """
    Inverse Jacobian for linear interval Lagrangian element.

    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    xe : np.ndarray
        coordinates of element nodes shape (nels,2,1). nels must be either 1,
        ncoords/2 or the same as ncoords. The two exceptions are if
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
    Jinv : np.ndarray, shape (ncoords,1,1) or (nels,1,1)
           Jacobian.
    detJ : np.ndarray, shape (ncoords) or (nels)
           if return_det is True, determinant of Jacobian.

    """
    # jacobian
    J = jacobian(xi=xi,xe=xe,all_elems=all_elems)
    # raise warning if determinant close to zero
    if np.any(np.isclose(J, 0)):
        warn("Determinant of element numerically close to zero.")
    elif np.any(J<0):
        raise ValueError("Determinant of Jacobian negative.")
    # return inverse
    if not return_det:
        return (J**(-1))[:,None,None]
    else:
        return (J**(-1))[:,None,None], J

def bmatrix(xi: np.ndarray,
            xe: np.ndarray,
            all_elems: bool = False, 
            return_detJ: bool = False,
            **kwargs: Any) -> np.ndarray:
    """
    B matrix for linear interval Lagrangian element to calculate
    to calculate strains, stresses etc. from nodal values

    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    xe : np.ndarray
        coordinates of element nodes shape (nels,2,1). nels must be either 1,
        ncoords/2 or the same as ncoords. The two exceptions are if
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
    B : np.ndarray, shape (ncoords,1,2) or (nels,1,2)
        B matrix.
    detJ : np.ndarray, shape (ncoords) or (nels)
           determinant of Jacobian.

    """
    # check coordinates and node data for consistency
    xe,xi,_,_ = check_inputs(xi=xi,xe=xe,all_elems=all_elems)
    # collect inverse jacobians
    if not return_detJ:
        invJ = invjacobian(xi=xi,xe=xe,
                           return_det=return_detJ)
    else:
        invJ,detJ = invjacobian(xi=xi,xe=xe,
                                return_det=return_detJ)
    # helper array to collect shape function derivatives
    B = shape_functions_dxi(xi)*invJ
    if not return_detJ:
        return B
    else:
        return B, detJ
