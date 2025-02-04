from warnings import warn

import numpy as np

def create_edofMat(nelx,nely,nnode_dof):
    """
    Create element degree of freedom matrix for bilinear elements in a regular
    mesh.
    
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
    n1 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    n2 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    """
    # create arrays for indexing
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    # 
    edofMat = np.repeat(np.column_stack((n1+1, n2+1, n2, n1))*nnode_dof,nnode_dof,axis=1) 
    edofMat = edofMat + np.tile(np.arange(nnode_dof),4)[None,:]
    return edofMat, n1, n2

def update_indices(indices,fixed,mask):
    """
    Update the indices for the stiffness matrix construction by kicking out
    the fixed degrees of freedom and renumbering the indices. This is useful 
    only if just one set of boundary conditions need to be solved.

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

def shape_functions_bilinquad(xi,eta):
    """
    Shape functions for bilinear quadrilateral Lagrangian element in reference 
    domain. Coordinates bounded in [-1,1].
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    shape_functions : np.ndarray, shape (4)
        values of shape functions at specified coordinate(s).
        
    """
    return 1/4 * np.array([(1-xi)*(1-eta),
                           (1+xi)*(1-eta),
                           (1+xi)*(1+eta),
                           (1-xi)*(1+eta)])

def shape_functions_dxi_bilinquad(xi,eta):
    """
    Gradient of shape functions for bilinear quadrilateral Lagrangian element. 
    The derivative is taken with regards to the reference coordinates, not the 
    physical coordinates.
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    shape_functions_dxi : np.ndarray, shape (4,2)
        gradient of shape functions at specified coordinate(s).
        
    """
    return 1/4 * np.array([[(-1)*(1-eta),(-1)*(1-xi)],
                           [(1-eta),(-1)*(1+xi)],
                           [1+eta,1+xi],
                           [(-1)*(1+eta),1-xi]])

def jacobian_bilinquad(xe,xi,eta):
    """
    Jacobian for quadratic bilinear Lagrangian element. 
    
    Parameters
    ----------
    xe : np.ndarray
        coordinates of element nodes shape (nels,4,2). Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    J : np.ndarray, shape (4,4)
        Jacobian.
        
    """
    return shape_functions_dxi_bilinquad(xi,eta).T @ xe 

def invjacobian_bilinquad(xe,xi,eta):
    """
    Inverse Jacobian for bilinear quadrilateral Lagrangian element. 
    
    Parameters
    ----------
    xe : np.ndarray
        coordinates of element nodes shape (nels,4,2). Please look at the 
        definition/function of the shape function, then the node ordering is 
        clear.
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    J : np.ndarray, shape (4,4)
        Jacobian.
        
    """
    # jacobian
    J = jacobian_bilinquad(xe,xi,eta)
    # determinant
    det = (J[:,0,0]*J[:,1,1]) - (J[:,1,0]*J[:,0,1])
    # raise warning if determinant close to zero
    if np.any(np.isclose(det, 0)):
        warn("Determinant of element numerically close to zero.")
    elif np.any(J<0):
        raise ValueError("Determinant of Jacobian negative.")
    # adjungate matrix
    adj = np.empty_like(J)
    adj[:, 0, 0], adj[:, 1, 1] = J[:, 1, 1], J[:, 0, 0]
    adj[:, 0, 1], adj[:, 1, 0] = -J[:, 0, 1], -J[:, 1, 0]
    # return inverse
    return adj/det

def jacobian_bilinquad_rectangle(a,b):
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
    J : np.ndarray, shape (4,4)
        Jacobian.
        
    """
    return 1/2 * np.array([[a,0],[0,b]])

def invjacobian_bilinquad_rectangle(a,b):
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
    J : np.ndarray, shape (4,4)
        Jacobian.
        
    """ 
    return 2 * np.array([[1/a,0],[0,1/b]])

def interpolate_2d(ue,xi,eta,
                   shape_functions=shape_functions_bilinquad):
    """
    Interpolate state variable in each element. Coordinates are assumed to be
    in the reference domain.
    
    Parameters
    ----------
    ue : np.ndarray
        shape (nels,nedof).
    xi : np.ndarray
        x coordinate of shape (nels). Coordinates are assumed to be
        in the reference domain.
    eta : np.ndarray
        y coordinate of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    u : np.ndarray, shape (nels,nnodedof)
        interpolated state variable.
        
    """
    # interpolate
    interpolation = shape_functions(xi,eta)
    # get parameters for reshaping to desired end shape
    nshapef = interpolation.shape[1] 
    nnodedof = int(ue.shape[1]/nshapef)
    u = ue * np.repeat(interpolation, nnodedof)[None,:]
    u = u.dot(np.tile(np.eye(nnodedof),(nshapef,1)))
    return u
