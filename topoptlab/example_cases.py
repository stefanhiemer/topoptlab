import numpy as np

def mbb_2d(nelx,nely,ndof,**kwargs):
    """
    This is the standard case from the 88 line code paper.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix x displacements to zero)
    fixed = np.hstack((np.arange(0,2*(nely+1),2), # symmetry 
                       np.array([ndof-1]))) # fixation bottom right
    # force pushing down at left top
    f[1, 0] = -1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def mbb_3d(nelx,nely,nelz,ndof,**kwargs):
    """
    This is the standard case from the 88 line code paper.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix x displacements to zero)
    fixed = np.hstack((np.arange(0,3*(nelx+1)*(nely+1),3), # symmetry in x 
                       np.arange((nelx+1)*(nely+1)*3,
                                 ndof+1,
                                 (nelx+1)*(nely+1)*3),) ) # fixation bottom right
    # force pushing down in y direction on top of symmetry plane
    f[np.arange(1,(nelx+1)*(nely+1)*(nelz+1)*3,(nelx+1)*(nely+1)*3), 0] = -1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def cantilever_2d(nelx,nely,ndof,**kwargs):
    """
    This is the corrected example 5.1 from the 88 line code paper.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # fix all dofs at left wall
    fixed = np.arange(0,2*(nely+1))
    # force at cantilever tip located at bottom
    f[-1,0] = -1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def cantilever_2d_twoloads(nelx,nely,ndof,**kwargs):
    """
    This is the corrected example 5.2 from the 88 line code paper.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # fix all dofs at left wall
    fixed = np.arange(0,2*(nely+1))
    # force at cantilever tip located at bottom
    f[-1,0] = -1
    # force at cantilever tip located at top
    f[2*nelx*(nely+1)+1,1] = 1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def cantilever_2d_twoloads_wrong(nelx,nely,ndof,**kwargs):
    """
    This is the example 5.2 from the 88 line code paper. It is actually wrong
    as it misses out on fixing the last dof in y direction on the left wall. It
    is here purely for testing.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # fix all dofs at left wall
    fixed = np.arange(0,2*nely+1)
    # force at cantilever tip located at bottom
    f[-1,0] = -1
    # force at cantilever tip located at top
    f[2*nelx*(nely+1)+1,1] = 1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def cantilever_2d_wrong(nelx,nely,ndof,**kwargs):
    """
    This is the example 5.1 from the 88 line code paper. It is actually wrong
    as it misses out on fixing the last dof in y direction on the left wall. It
    is here purely for testing.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # fix all dofs at left wall
    fixed = np.arange(0,2*nely+1)
    # force at cantilever tip located at bottom
    f[-1,0] = -1
    return u,f,fixed,np.setdiff1d(dofs, fixed)

def heatplate_2d(nelx,nely,ndof,**kwargs):
    """
    Heat conduction problem with an evenly heated plate attached to a heat 
    sink at the negative x side.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.

    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be 
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # heat sink
    fixed = np.arange(int(nely / 2 + 1 - nely / 20), 
                      int(nely / 2 + 1 + nely / 20) + 1)
    # load/source
    f[:, 0] = -1 # constant source
    return u,f,fixed,np.setdiff1d(dofs, fixed)