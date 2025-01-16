import numpy as np

def create_edofMat(nelx,nely,nnode_dof):
    """
    Create element degree of freedom matrix for bilinear elements.
    
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
    the fixed degrees of freedom and renumbering the indices.

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

def lk_linear_elast_2d(E=1,nu=0.3):
    """
    Create element stiffness matrix for 2D isotropic linear elasticity with 
    bilinear quadratic elements.
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson' ratio.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    Ke = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return Ke

def lk_poisson_2d():
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadratic elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = np.array([[2/3, -1/6, -1/3, -1/6,],
                   [-1/6, 2/3, -1/6, -1/3],
                   [-1/3, -1/6, 2/3, -1/6],
                   [-1/6, -1/3, -1/6, 2/3]])
    return Ke

def l_screened_poisson_2d(rmin):
    """
    Create matrix for 2D screened Poisson equation with bilinear quadratic 
    elements. Taken from the 88 lines code and slightly modified.
    
    Parameters
    ----------
    rmin : float
        filter radius.
        
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    Ke = (rmin**2) * np.array([[2/3, -1/6, -1/3, -1/6],
                               [-1/6, 2/3, -1/6, -1/3],
                               [-1/3, -1/6, 2/3, -1/6],
                               [-1/6, -1/3, -1/6, 2/3]]) + \
                     np.array([[1/9, 1/18, 1/36, 1/18],
                               [1/18, 1/9, 1/18, 1/36],
                               [1/36, 1/18, 1/9, 1/18],
                               [1/18, 1/36, 1/18, 1/9]])
    return Ke

def shape_functions_bilinquad(x,y):
    """
    Shape functions for quadratic bilinear element. Coordinates bounded in 
    [-1,1].
    
    Parameters
    ----------
    x : np.ndarray
        x coordinate of shape (ncoords).
    y : np.ndarray
        y coordinate of shape (ncoords).
        
    Returns
    -------
    shape_functions : np.ndarray, shape (4)
        element stiffness matrix.
        
    """
    return 1/4 * np.array([(1-x)*(1-y),
                           (1+x)*(1-y),
                           (1+x)*(1+y),
                           (1-x)*(1+y)])

def interpolate_2d(ue,x,y,
                   shape_functions=shape_functions_bilinquad):
    """
    Interpolate state variable in each element.
    
    Parameters
    ----------
    ue : np.ndarray
        shape (nels,nedof).
    x : np.ndarray
        x coordinate of shape (nels).
    y : np.ndarray
        y coordinate of shape (ncoords).
        
    Returns
    -------
    u : np.ndarray, shape (nels,nnodedof)
        interpolated state variable.
        
    """
    interpolation = shape_functions(x,y)
    nshapef = interpolation.shape[1] 
    nnodedof = int(ue.shape[1]/nshapef)
    u = ue * np.repeat(interpolation, nnodedof)[None,:]
    u = u.dot(np.tile(np.eye(nnodedof),(nshapef,1)))
    return u