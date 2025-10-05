# SPDX-License-Identifier: GPL-3.0-or-later
from warnings import warn

import numpy as np

def create_edofMat(nelx,nely,nelz,nnode_dof,dtype=np.int32):
    """
    Create element degree of freedom matrix for trilinear elements in a regular
    mesh.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    nnode_dof : int
        number of node degrees of freedom.    
    dtype : type
        data type for edofMat

    Returns
    -------
    edofMat : np.ndarray
        element degree of freedom matrix
    n1 : np.ndarray
        index array to help constructing the stiffness matrix.
    n2 : np.ndarray
        index array to help constructing the stiffness matrix.
    n3 : np.ndarray
        index array to help constructing the stiffness matrix.
    n4 : np.ndarray
        index array to help constructing the stiffness matrix.
    """
    # create arrays for indexing
    elx = np.arange(nelx, dtype=dtype)[None,:,None]
    ely = np.arange(nely, dtype=dtype)[None,None,:]
    elz = np.arange(nelz, dtype=dtype)[:,None,None]
    n1 = ((nelx+1)*(nely+1)*elz + (nely+1)*elx + ely).flatten()
    n2 = ((nelx+1)*(nely+1)*elz + (nely+1)*(elx+1) + ely).flatten()
    n3 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*elx + ely).flatten()
    n4 = ((nelx+1)*(nely+1)*(elz+1) + (nely+1)*(elx+1) + ely).flatten()
    # 
    edofMat = np.column_stack((n1+1,n2+1,n2,n1,
                               n3+1,n4+1,n4,n3))*nnode_dof
    edofMat = np.repeat(edofMat,nnode_dof,axis=1)
    edofMat = edofMat + np.tile(np.arange(nnode_dof,dtype=dtype),8)[None,:]
    return edofMat, n1, n2, n3, n4

def apply_pbc(edofMat,pbc,nelx,nely,nelz,nnode_dof,
              dtype=np.int32,**kwargs):
    """
    Convert a given element-degree-of-freedom matrix (edofMat) of a regular 
    mesh of first order Lagrangian hexahedral elements with free 
    boundary conditions to the corresponding edofMat with periodic boundary 
    conditions by re-labelling the nodal degrees of freedom
    
    Parameters
    ----------
    edofMat : np.ndarray of shape (nel,8*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular 
        mesh of first order Lagrangian quadrilateral elements with free 
        boundary conditions.
    pbc : list or np.ndarray of shape/len 2 with elements of datatype bool
        periodic boundary condition flags.
    nelx : int
        number of elements in x-direction.
    nely : int
        number of elements in y-direction.
    nelz : int
        number of elements in z-direction.
    nnode_dof : int
        number of degrees of freedom per node.
    dtype : type
        datatype of element degrees of freedom. Should be just large enough to 
        store the highest number of the degrees of freedom to save memory. For
        practical purposes np.int32 should do the job.
        
    Returns
    -------
    edofMat_new : np.ndarray of shape (nel,8*nnode_dof)
        element-degree-of-freedom matrix (edofMat) of a regular 
        mesh of first order Lagrangian hexahedral elements with periodic 
        boundary conditions with re-labelled nodal degrees of freedom.
    """
    # update indices
    # x
    if pbc[0] and not pbc[1]:
        edofMat -= np.floor(edofMat/((nelx+1)*(nely+1)*nnode_dof))\
                   .astype(dtype)*(nely+1)*nnode_dof
    # y
    elif not pbc[0] and pbc[1]:
        edofMat -= np.floor(edofMat/((nely+1)*nnode_dof))\
                   .astype(dtype)*nnode_dof
    # x and y 
    elif pbc[0] and pbc[1]:
        edofMat -= (np.floor(edofMat/((nely+1)*nnode_dof)).astype(dtype)+\
                   np.floor(edofMat/((nelx+1)*(nely+1)*nnode_dof))\
                   .astype(dtype)*nely) * nnode_dof
    # only z, no updates needed
    elif not pbc[0] and not pbc[1] and pbc[2]:
        pass
    # no pbc
    else:
        return edofMat
    # reassign indices
    nel = nelx*nely*nelz
    # x
    if pbc[0]:
        # find original and periodic elements
        n_xy = nelx*nely
        org = np.arange(nely)[None,:]+np.arange(nelz)[:,None]*n_xy
        org = org.flatten()
        pbc_x = np.arange(n_xy-nely,n_xy)[None,:]+np.arange(nelz)[:,None]*n_xy
        pbc_x = pbc_x.flatten()
        # reassign indices
        edofMat[pbc_x,nnode_dof:2*nnode_dof] = edofMat[org,:nnode_dof]
        edofMat[pbc_x,2*nnode_dof:3*nnode_dof] = edofMat[org,3*nnode_dof:4*nnode_dof]
        edofMat[pbc_x,5*nnode_dof:6*nnode_dof] = edofMat[org,4*nnode_dof:5*nnode_dof]
        edofMat[pbc_x,6*nnode_dof:7*nnode_dof] = edofMat[org,-nnode_dof:]
    # y
    if pbc[1]:
        # find original and periodic elements
        org = np.arange(0,nel,nely)
        pbc_y = np.arange(nely-1,nel+1,nely)
        # reassign indices
        edofMat[pbc_y,:nnode_dof] = edofMat[org,3*nnode_dof:4*nnode_dof]
        edofMat[pbc_y,nnode_dof:2*nnode_dof] = edofMat[org,2*nnode_dof:3*nnode_dof]
        edofMat[pbc_y,4*nnode_dof:5*nnode_dof] = edofMat[org,-nnode_dof:]
        edofMat[pbc_y,5*nnode_dof:6*nnode_dof] = edofMat[org,6*nnode_dof:7*nnode_dof]
    # z
    if pbc[2]:
        # find original and periodic elements
        org = np.arange(nelx*nely)
        pbc_z = np.arange(nel-nelx*nely,nel)
        # reassign indices
        edofMat[pbc_z,4*nnode_dof:] = edofMat[org,:4*nnode_dof]
    return edofMat

def check_inputs(xi,eta,zeta,xe=None,all_elems=False,**kwargs):
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
        coordinates of element nodes shape (nels,8,3). nels must be either 1, 
        ncoords/8 or the same as ncoords. The two exceptions are if 
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
    xe : np.ndarray
        coordinates of element nodes shape (n,8,2).
    xi : float or np.ndarray
        x coordinate in the reference domain of shape (n).
    eta : float or np.ndarray
        y coordinate in the reference domain of shape (n). 
    zeta : np.ndarray
        z coordinate in the reference domain of shape (n).
    """
    #
    if isinstance(xi,np.ndarray) and isinstance(eta,np.ndarray):
        #
        if len(xi.shape) != 1 or len(eta.shape) != 1 or len(zeta.shape) != 1:
            raise ValueError("xi and eta must be 1D: ", 
                             xi.shape,eta.shape)
        elif any([xi.shape[0] != eta.shape[0],
                  xi.shape[0] != zeta.shape[0],
                  eta.shape[0] != zeta.shape[0]]):
            raise ValueError("xi, eta and zeta must have same shape: ", 
                             xi.shape,eta.shape,zeta.shape)
        else:
            ncoords = xi.shape[0]
    elif (isinstance(xi,int) and isinstance(eta,int) and \
          isinstance(eta,int)) or\
         (isinstance(xi,float) and isinstance(eta,float) and \
          isinstance(zeta,float)):
        ncoords = 1
    else:
        raise ValueError("Datatypes of xi, eta and zeta inconsistent.")
    if xe is not None:
        if len(xe.shape) == 2:
            xe = xe[None,:,:]
        if xe.shape[-2:] != (8,3):
            raise ValueError("shapes of xe must be (nels,8,3) or (8,3).")
        nels = xe.shape[0]
        if not all_elems and all([nels != ncoords,8*nels != ncoords,
                                  nels != 1,ncoords!=1]):
            raise ValueError("shapes of nels and ncoords incompatible.")
        elif all_elems:
            xi = np.tile(xi,nels)
            eta = np.tile(eta,nels)
            zeta = np.tile(zeta,nels)
            xe = np.repeat(xe,repeats=ncoords,axis=0)
        elif 8*nels == ncoords:
            xe = np.repeat(xe,repeats=8,axis=0)
        return xe,xi,eta,zeta
    else:
        return ncoords

def shape_functions(xi,eta,zeta,**kwargs):
    """
    Shape functions for trilinear hexahedron Lagrangian element in reference 
    domain. Coordinates bounded in [-1,1].
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    shape_functions : np.ndarray, shape (ncoords,8)
        values of shape functions at specified coordinate(s).
        
    """
    return 1/8 * np.column_stack(((1-xi)*(1-eta)*(1-zeta),
                                  (1+xi)*(1-eta)*(1-zeta),
                                  (1+xi)*(1+eta)*(1-zeta),
                                  (1-xi)*(1+eta)*(1-zeta),
                                  (1-xi)*(1-eta)*(1+zeta),
                                  (1+xi)*(1-eta)*(1+zeta),
                                  (1+xi)*(1+eta)*(1+zeta),
                                  (1-xi)*(1+eta)*(1+zeta)))

def shape_functions_dxi(xi,eta,zeta,**kwargs):
    """
    Gradient of shape functions for trilinear hexahedron Lagrangian element. 
    The derivative is taken with regards to the reference coordinates, not the 
    physical coordinates.
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
        
    Returns
    -------
    shape_functions_dxi : np.ndarray, shape (ncoords,8,3)
        gradient of shape functions at specified coordinate(s).
        
    """
    dx = 1/8 * np.column_stack(((-1)*(1-eta)*(1-zeta),
                                (-1)*(1-xi)*(1-zeta),
                                (1-xi)*(1-eta)*(-1),
                                (1-eta)*(1-zeta),
                                (-1)*(1+xi)*(1-zeta),
                                (1+xi)*(1-eta)*(-1),
                                (1+eta)*(1-zeta),
                                (1+xi)*(1-zeta),
                                (1+xi)*(1+eta)*(-1),
                                (-1)*(1+eta)*(1-zeta),
                                (1-xi)*(1-zeta),
                                (1-xi)*(1+eta)*(-1),
                                (-1)*(1-eta)*(1+zeta),
                                (-1)*(1-xi)*(1+zeta),
                                (1-xi)*(1-eta),
                                (1-eta)*(1+zeta),
                                (-1)*(1+xi)*(1+zeta),
                                (1+xi)*(1-eta),
                                (1+eta)*(1+zeta),
                                (1+xi)*(1+zeta),
                                (1+xi)*(1+eta),
                                (-1)*(1+eta)*(1+zeta),
                                (1-xi)*(1+zeta),
                                (1-xi)*(1+eta)))
    return dx.reshape(int(np.prod(dx.shape)/24),8,3)

def jacobian(xi,eta,zeta,xe,all_elems=False,**kwargs):
    """
    Jacobian for quadratic bilinear Lagrangian element. 
    
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
        coordinates of element nodes shape (nels,8,3). nels must be either 1, 
        ncoords/8 or the same as ncoords. The two exceptions are if 
        ncoords = 1 or all_elems is True. 
        Please look at the definition/function of the shape function, then the 
        node ordering is clear.
    all_elems : bool
        if True, coordinates are evaluated for all elements. Useful for 
        creating elements etc.
        
    Returns
    -------
    J : np.ndarray, shape (ncoords,3,3) or (nels,3,3)
        Jacobian.
        
    """
    # check coordinates and node data for consistency
    xe,xi,eta,zeta = check_inputs(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                  all_elems=all_elems)
    return shape_functions_dxi(xi=xi,eta=eta,zeta=zeta).transpose([0,2,1]) @ xe

def invjacobian(xi,eta,zeta,xe,
                all_elems=False,return_det=False):
    """
    Inverse Jacobian for bilinear quadrilateral Lagrangian element. 
    
    Parameters
    ----------
    xi : np.ndarray
        x coordinate in the reference domain of shape (ncoords).
    eta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray
        z coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,8,2). nels must be either 1, 
        ncoords/8 or the same as ncoords. The two exceptions are if 
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
    Jinv : np.ndarray, shape (ncoords,3,3) or (nels,3,3)
           Jacobian.
        
    """
    # jacobian
    J = jacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,all_elems=all_elems)
    # determinant
    det = (J[:,0,0]*(J[:,1,1]*J[:,2,2] - J[:,1,2]*J[:,2,1])-
           J[:,0,1]*(J[:,1,0]*J[:,2,2] - J[:,1,2]*J[:,2,0])+
           J[:,0,2]*(J[:,1,0]*J[:,2,1] - J[:,1,1]*J[:,2,0]))
    # raise warning if determinant close to zero
    if np.any(np.isclose(det, 0)):
        warn("Determinant of element numerically close to zero.")
    elif np.any(det<0):
        raise ValueError("Determinant of Jacobian negative.")
    # adjungate matrix
    adj = np.empty_like(J)
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
        return adj/det[:,None,None]
    else:
        return adj/det[:,None,None], det

def jacobian_cuboid(a,b,c):
    """
    Jacobian for cuboid quadratic bilinear Lagrangian element. 
    
    Parameters
    ----------
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.
    c : float
        length of rectangle in z direction.
        
    Returns
    -------
    J : np.ndarray, shape (3,3)
        Jacobian.
        
    """
    return 1/2 * np.array([[a,0,0],[0,b,0],[0,0,c]])

def invjacobian_cuboid(a,b,c):
    """
    Inverse Jacobian for cuboid quadratic bilinear Lagrangian element. 
    
    Parameters
    ----------
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.
    c : float
        length of rectangle in z direction.
        
    Returns
    -------
    J : np.ndarray, shape (3,3)
        Jacobian.
        
    """ 
    return 2 * np.array([[1/a,0,0],[0,1/b,0],[0,0,1/c]])

def bmatrix(xi,eta,zeta,xe,
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
    zeta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    xe : np.ndarray
        coordinates of element nodes shape (nels,8,3). nels must be either 1, 
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
    B : np.ndarray, shape (ncoords,9,24) or (nels,9,24)
        B matrix.
        
    """
    # check coordinates and node data for consistency
    xe,xi,eta,zeta = check_inputs(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                  all_elems=all_elems)
    # collect inverse jacobians
    invJ = invjacobian(xi,eta,zeta,xe)
    if not return_detJ:
        invJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                           return_det=return_detJ)
    else:
        invJ,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,xe=xe,
                                return_det=return_detJ)
    # helper array to collect shape function derivatives
    helper = np.zeros((invJ.shape[0],9,24)) # shape (nel,9,24)
    shp = shape_functions_dxi(xi,eta,zeta).transpose([0,2,1])
    helper[:,:3,::3] = shp
    helper[:,3:6,1::3] = shp.copy() # copy to avoid np.views
    helper[:,6:,2::3] = shp.copy() # copy to avoid np.views
    #
    B = np.array([[1,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,1,0,1,0],
                  [0,0,1,0,0,0,1,0,0],
                  [0,1,0,1,0,0,0,0,0]])@np.kron(np.eye(3),invJ)@helper
    if not return_detJ:
        return B
    else:
        return B, detJ

def bmatrix_cuboid(xi,eta,zeta,a,b,c):
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
    zeta : np.ndarray
        y coordinate in the reference domain of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    a : float
        length of rectangle in x direction.
    b : float
        length of rectangle in y direction.
    c : float
        length of rectangle in z direction.
        
    Returns
    -------
    B : np.ndarray, shape (ncoords,9,24)
        B matrix.
        
    """
    # check coordinates for consistency
    ncoords = check_inputs(xi,eta,zeta) 
    # collect inverse jacobians
    invJ = invjacobian_cuboid(a,b,c)
    # helper array to collect shape function derivatives
    helper = np.zeros((ncoords,9,24))
    shp = shape_functions_dxi(xi,eta,zeta).transpose([0,2,1])
    helper[:,:3,::3] = shp
    helper[:,3:6,1::3] = shp.copy() # copy to avoid np.views
    helper[:,6:,2::3] = shp.copy() # copy to avoid np.views
    #
    B = np.array([[1,0,0,0,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0],
                  [0,0,0,0,0,0,0,0,1],
                  [0,0,0,0,0,1,0,1,0],
                  [0,0,1,0,0,0,1,0,0],
                  [0,1,0,1,0,0,0,0,0]])@np.kron(np.eye(3),invJ)@helper
    return B

