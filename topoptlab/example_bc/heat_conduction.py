# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple
import numpy as np

def heatplate_2d(nelx: int, nely: int,
                 ndof: int, **kwargs: Any
                 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    Heat conduction problem with an evenly heated plate attached to a heat 
    sink at the negative x side. Example case taken from the standard TO 
    textbook by Sigmund and Bendsoe page 271.
    
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
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

def rectangle_2d(nelx: int, nely: int,
                 ndof: int, **kwargs: Any
                 ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,None]:
    """
    Heat conduction through rectangular beam with heat source located at 
    left side and heat sink at right side.
    
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

    """
    # BC's
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # heat sink
    fixed = np.arange(nely+1)
    # load/source
    f[-(nely+1):, 0] = 1 # constant source
    return u,f,fixed,np.setdiff1d(dofs,fixed),None