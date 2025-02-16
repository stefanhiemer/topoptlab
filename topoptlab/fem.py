from itertools import product

import numpy as np

from topoptlab.elements.bilinear_quadrilateral import shape_functions

def create_matrixinds(edofMat,mode="full"):

    #
    ne = edofMat.shape[1]
    if mode == "full":
        iM = np.tile(edofMat,ne)
        jM = np.repeat(edofMat,ne)
    elif mode == "half":
        iM = [edofMat[:,i:] for i in np.arange(ne)]
        iM = np.column_stack(iM)
        jM = np.repeat(edofMat,np.arange(ne,0,-1),axis=1)
    return iM.flatten(),jM.flatten()

def update_indices(indices,fixed,mask):
    """
    Update the indices for the stiffness matrix construction by kicking out
    the fixed degrees of freedom and renumbering the indices. This is useful
    only if just one set of boundary conditions needs to be solved.

    Parameters
    ----------
    indices : np.array
        indices of degrees of freedom used to construct the stiffness matrix.
    fixed : np.array
        indices of fixed degrees of freedom.
    mask : np.array
        mask to kick out fixed degrees of freedom.

    Returns
    -------
    indices : np.ndarray
        updated indices.

    """
    val, ind = np.unique(indices,return_inverse=True)

    _mask = ~np.isin(val, fixed)
    val[_mask] = np.arange(_mask.sum())

    return val[ind][mask]

def interpolate(ue,xi,eta,zeta=None,
                shape_functions=shape_functions):
    """
    Interpolate node values in each element. Coordinates are assumed to be
    in the reference domain.

    Parameters
    ----------
    ue : np.ndarray,shape (nels,nedof).
        node values used for interpolation.
    xi : np.ndarray
        x coordinate of shape (nels). Coordinates are assumed to be
        in the reference domain.
    eta : np.ndarray
        y coordinate of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray
        z coordinate of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    shape_functions : callable
        shape functions of respective element
    Returns
    -------
    u : np.ndarray, shape (nels,nnodedof)
        interpolated state variable.

    """
    # interpolate
    if zeta is None:
        interpolation = shape_functions(xi,eta)
    else:
        interpolation = shape_functions(xi,eta,zeta)
    # get parameters for reshaping to desired end shape
    nshapef = interpolation.shape[1]
    nnodedof = int(ue.shape[1]/nshapef)
    u = ue * np.repeat(interpolation, nnodedof)[None,:]
    u = u.dot(np.tile(np.eye(nnodedof),(nshapef,1)))
    return u

def get_integrpoints(ndim,nq,method):
    """
    Get integration points and weights for numerical quadrature of integrals in
    interval [-1,1].

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    nq : int
        number of integration/quadrature points.
    method : str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Currently only 'gauss-legendre'
        supported as str.

    Returns
    -------
    x : np.ndarray, shape (nq,ndim)
        coordinates of quadrature point.
    w : np.ndarray, shape (nq)
        weights of quadrature point.

    """
    if method == "gauss-legendre":
        x,w = np.polynomial.legendre.leggauss(nq)
    elif hasattr(method, '__call__'):
        x,w = method(nq)
    else:
        raise ValueError("Invalid quadrature method.")
    # generate grid of points
    x = np.array(list(product(x,repeat=ndim)))
    w = np.prod(np.array(list(product(w,repeat=ndim))),axis=1)
    return x,w
