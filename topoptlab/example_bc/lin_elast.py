# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np

def mbb_2d(nelx: int, nely: int,
           ndof: int, **kwargs: Any
           ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
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
    springs : None
        example has no springs.

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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def mbb_3d(nelx: int, nely: int, nelz: int,
           ndof: int, **kwargs: Any
           ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    This is an equivalent to the standard case from the 88 line code paper.

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
    springs : None
        example has no springs.
    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix x displacements to zero)
    xsymmetry = np.arange(0,3*(nely+1),3)
    xsymmetry = np.tile(xsymmetry,nelz+1)+\
                np.repeat(np.arange(0,ndof,3*(nelx+1)*(nely+1)),nely+1)
    # symmetry bc (fix z displacements to zero)
    zsymmetry = np.arange(2,(nelx+1)*(nely+1)*3,3)
    # fix y dofs at support position
    fixation = np.arange((nelx+1)*(nely+1)*3 - 2,ndof,(nelx+1)*(nely+1)*3)
    #
    fixed = np.hstack((xsymmetry, zsymmetry,
                       fixation,
                       fixation+1)) # z fixation
    # force pushing down in y direction on top of symmetry plane
    f[np.arange(1,(nelx+1)*(nely+1)*(nelz+1)*3,(nelx+1)*(nely+1)*3), 0] = -1
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def cantilever_2d(nelx: int, nely: int,
                  ndof: int, **kwargs: Any
                  ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
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
    springs : None
        example has no springs.

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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def cantilever_2d_twoloads(nelx: int, nely: int,
                           ndof: int, **kwargs: Any
                           ) -> Tuple[np.ndarray,np.ndarray,
                                      np.ndarray,np.ndarray,None]:
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
    springs : None
        example has no springs.
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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def cantilever_2d_twoloads_wrong(nelx: int, nely: int,
                                 ndof: int, **kwargs: Any
                                 ) -> Tuple[np.ndarray,np.ndarray,
                                            np.ndarray,np.ndarray,None]:
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
    springs : None
        example has no springs.

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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def cantilever_2d_wrong(nelx: int, nely: int,
                        ndof: int, **kwargs: Any
                        ) -> Tuple[np.ndarray,np.ndarray,
                                   np.ndarray,np.ndarray,None]:
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
    springs : None
        example has no springs.
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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def cantilever_3d(nelx: int, nely: int, nelz: int,
                  ndof: int,
                  force_mode: str = "linear",
                  **kwargs: Any
                  ) -> Tuple[np.ndarray,np.ndarray,
                             np.ndarray,np.ndarray,None]:
    """
    This is an equivalent to 4.2 in 
    
    Amir, Oded, Niels Aage, and Boyan S. Lazarov. "On multigrid-CG for 
    efficient topology optimization." Structural and Multidisciplinary 
    Optimization 49.5 (2014): 815-829.

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
    force_mode : str
        mode how to distribute forces at cantilever end.

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
    springs : None
        example has no springs.
    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # fixation to wall
    fixation = np.arange(0,3*(nely+1),3)
    fixation = np.tile(fixation,nelz+1)+\
              np.repeat(np.arange(0,ndof,3*(nelx+1)*(nely+1)),nely+1)
    fixation = np.hstack( (fixation,fixation+1,fixation+2) )
    # symmetry bc (fix z displacements to zero)
    zsymmetry = np.arange(2,(nelx+1)*(nely+1)*3,3)
    #
    fixed = np.union1d(zsymmetry,
                        fixation)
    # force pushing down in y direction on end of cantilever
    if force_mode == "linear":
        f0 = -np.linspace(1.,0.,nelz+1)
    elif force_mode == "constant":
        f0 = -1. 
    else:
        raise ValueError("force_mode must be either 'linear' or 'constant': ",
                         force_mode)
    f[np.arange((nelx+1)*(nely+1)*3 - 2,ndof,(nelx+1)*(nely+1)*3), 0] = f0
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def forceinverter_2d(nelx: int, nely: int,
                     ndof: int, **kwargs: Any
                     ) -> Tuple[np.ndarray,np.ndarray,
                                np.ndarray,np.ndarray,None]:
    """
    Force inverter as example case for compliant mechanisms. Example case taken 
    from the standard TO textbook by Sigmund and Bendsoe page 269.

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
    springs : list
        contains two 1D np.ndarrays of equal length. first is of integer type
        and contains the indices of dofs attached to a spring. second contains
        the spring constants.

    """
    # force and displacements
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    #
    fixed = np.union1d(np.arange(1,(nelx+1)*(nely+1)*2,(nely+1)*2), # symmetry
                       np.arange(2*(nely+1)-4,2*(nely+1))) # bottom left fixation
    # load/source
    f[0,0] = 1
    #
    springs = [np.array([0,2*nelx*(nely+1)]),np.array([0.1,0.1])]
    return u,f,fixed,np.setdiff1d(np.arange(ndof),fixed),springs

def forceinverter_3d(nelx: int, nely: int, nelz: int,
                     ndof: int, fixation_mode: str = "point",
                     **kwargs: Any
                     ) -> Tuple[np.ndarray,np.ndarray,
                                np.ndarray,np.ndarray,None]:
    """
    Force inverter case as in 
    
    Amir, Oded, Niels Aage, and Boyan S. Lazarov. "On multigrid-CG for 
    efficient topology optimization." Structural and Multidisciplinary 
    Optimization 49.5 (2014): 815-829.

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
    fixation_mode : str 
        either "line" or "point".

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
    springs : list
        contains two 1D np.ndarrays of equal length. first is of integer type
        and contains the indices of dofs attached to a spring. second contains
        the spring constants.

    """
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix y displacements to zero)
    ysymmetry = np.arange(1,ndof,3*(nely+1))
    # symmetry bc (fix z displacements to zero)
    zsymmetry = np.arange(2,(nelx+1)*(nely+1)*3,3)
    #
    if fixation_mode == "point": 
        fixation = np.array([ndof - 3])
    elif fixation_mode == "line": 
        fixation = np.arange( ndof-3, 3*(nelx+1)*(nely+1)*nelz,-3*(nely+1) )
    else:
        raise ValueError("fixation mode must be either 'line' or 'point': ",
                         fixation_mode)
    #
    fixed = np.hstack((ysymmetry,
                       zsymmetry,
                       fixation))
    # load/source
    f[0,0] = 1
    #
    springs = [np.array([0,3*nelx*(nely+1)]),np.array([0.1,1e-4])]
    return u,f,fixed,np.setdiff1d(dofs,fixed),springs

def threepointbending_2d(nelx: int, nely: int,
                         ndof: int, **kwargs: Any
                         ) -> Tuple[np.ndarray,np.ndarray,
                                    np.ndarray,np.ndarray,None]:
    """
    both displacement fixed at bottom left and bottom right and force pushes
    down on the middle top.

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
    springs : None
        example has no springs.

    """
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # heat sink
    fixed = np.hstack(([2*nely,2*(nely+1)-1], # bottom left
                       [ndof-2,ndof-1])) # bottom right
    # load/source
    f[nelx*(nely+1) + 1,0] = -1
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def xcenteredbeam_2d(nelx: int, nely: int,
                     ndof: int, **kwargs: Any
                     ) -> Tuple[np.ndarray,np.ndarray,
                                np.ndarray,np.ndarray,None]:
    """
    Both displacements fixed at the middle of the left and right boundary. No
    forces. This test case is mainly for cases where a force source due to
    another field (e. g. thermal stresses) appears.

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
    springs : None
        example has no springs.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # heat sink
    fixed = np.hstack(([0,2*nely], # xdofs fixed left side
                       [nely+1],
                       [2*nelx*(nely+1) + nely+1])) # bottom right
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def tensiletest_2d(nelx: int, nely: int,
                   ndof: int, ymirror: bool = True, 
                   **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    Tensile test with force applied in x direction with -x side fixed in terms
    of x dofs. BC details depend on options.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
    ymirror : bool
        if True, mirror axis at y=0 by setting the y dofs to zero. if False,
        then ydofs at x=0,y=Ly/2 and x=Lx,y=Ly/2

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
    springs : None
        example has no springs.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # applied force for tension
    f[-2:-(2*(nely+2)):-2] = 1.
    # fixed dofs
    # x mirror axis
    fixed = [np.arange(0,(nely+1)*2,2)]
    # y mirror axis
    if ymirror:
        fixed += [np.arange(1,ndof,2*(nely+1))]
    else:
        fixed += [nely+1,ndof-nely-1]
    fixed = np.hstack(fixed)
    #
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def tensiletest_3d(nelx: int, nely: int, nelz: int,
                   ndof: int, **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    x displacements fixed at left side and uniform force applied at right hand
    side. One y dof is fixed in the middle of the left side.

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
    springs : None
        example has no springs.
    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix x displacements to zero)
    xsymmetry = np.arange(0,3*(nely+1),3)
    xsymmetry = np.tile(xsymmetry,nelz+1)+\
                np.repeat(np.arange(0,ndof,3*(nelx+1)*(nely+1)),nely+1)
    # symmetry bc (fix y displacements to zero)
    ysymmetry = np.arange(1,ndof,3*(nely+1))
    # symmetry bc (fix z displacements to zero)
    zsymmetry = np.arange(2,(nelx+1)*(nely+1)*3,3)
    #
    fixed = np.hstack((xsymmetry,
                       ysymmetry,
                       zsymmetry))
    # force pulling
    tension = np.arange(0,3*(nely+1),3)
    tension = np.tile(tension,nelz+1)+\
              np.repeat(np.arange(0,ndof,3*(nelx+1)*(nely+1)),nely+1)
    tension = tension + (nely+1)*3*nelx
    f[tension,0] = 1.
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def selffolding_2d(nelx: int, nely: int,
                   ndof: int, **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,
                              np.ndarray,np.ndarray,
                              Tuple[np.ndarray,np.ndarray]]:
    """
    Symmetry axis on left side (x dofs fixed) and bottom node on left side has
    fixed y displacement as well. No forces applied as this is thought to be
    used with another physical phenomenon that induces forces by itself
    e. g. heat expansion.

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
    springs : tuple
        contains two 1D np.ndarrays of equal length. first is of integer type
        and contains the indices of dofs attached to a spring. second contains
        the spring constants.

    """
    if nely%2 !=0:
        raise ValueError("This example works only for nely equal to an even number.")
    # BC's
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], # symmetry
                       np.array([2*(nely+1)-1])) # bottom support
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    #
    springs = (np.array([2*nelx*(nely+1)+1]),
               np.array([5e-4]))
    return u,f,fixed,np.setdiff1d(dofs,fixed),springs

def selffolding_3d(nelx: int, nely: int, nelz: int,
                   ndof: int, **kwargs: Any
                   ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    Symmetry axis on left side (x dofs fixed) and bottom node on left side has
    fixed y displacement as well. No forces applied as this is thought to be
    used with another physical phenomenon that induces forces by itself
    e. g. heat expansion.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
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
        indices of free dofs shape (ndofs - nfixed).
    springs : None
        example has no springs.

    """
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # symmetry bc (fix x displacements to zero)
    xsymmetry = np.arange(0,3*(nely+1),3)
    xsymmetry = np.tile(xsymmetry,nelz+1)+\
                np.repeat(np.arange(0,ndof,3*(nelx+1)*(nely+1)),nely+1)
    # symmetry bc (fix z displacements to zero)
    zsymmetry = np.arange(2,(nelx+1)*(nely+1)*3,3)
    # fix y dofs at support position
    fixation = np.arange(1,ndof,(nelx+1)*(nely+1)*3)
    #
    fixed = np.hstack((xsymmetry,zsymmetry,
                       fixation,
                       fixation+1)) # z fixation
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def singlenode(nelx: int, nely: int,
               ndof: int, nelz: Union[None,int], 
               **kwargs: Any
               ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    Fix all dofs of first node. Typically used for homogenization or similar
    applications where the forces arise by another source (e. g. heat or via
    body forces).

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
    nelz : int or None
        number of elements in z direction.

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
        indices of free dofs shape (ndofs - nfixed).
    springs : None
        example has no springs.

    """
    if nelz is None:
        ndim=2
    else:
        ndim=3
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, int((ndim**2 + ndim) /2)))
    u = np.zeros((ndof, int((ndim**2 + ndim) /2)))
    # fix first node
    if nelz is None:
        fixed = dofs[:int(ndof/(nelx*nely))]
    else:
        fixed = dofs[:int(ndof/((nelx*nely*nelz)))]
    return u,f,fixed,np.setdiff1d(dofs,fixed),None
