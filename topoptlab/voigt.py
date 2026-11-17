# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

def from_voigt(A_v: np.ndarray, 
               eng_conv: bool = False) -> np.ndarray:
    """
    Convert 2nd rank tensor into from its Voigt representation to the standard
    matrix represenation.

    Parameters
    ----------
    A_v : np.ndarray
        2nd rank tensor in Voigt represenation (so a column vector) 
        shape (...,(ndim**2 + ndim) /2).
    eng_conv : bool 
        if True, engineering convention is applied meaning the shear components
        are scaled by a factor of two in Voigt notation. Usually applies only 
        to strains.

    Returns
    -------
    A : np.ndarray
        2nd rank tensor in matrix notation shape (...,ndim,ndim).
    """
    #
    l = A_v.shape[-1]
    #
    if l not in [1,3,6]:
        raise ValueError("This is not a vector compatible with the assumptions of Voigt representation.")
    #
    ndim = int(-1/2 + np.sqrt(1/2+2*l))
    #
    inds_v = np.array([[0,5,4],[5,1,3],[4,3,2]],
                      dtype=int)[:ndim,:ndim]%int((ndim**2 + ndim) /2)
    row = np.arange(ndim**2)%ndim
    col = np.floor_divide(np.arange(ndim**2,dtype=int),ndim)
    #
    A = np.zeros(A_v.shape[:-1]+tuple([ndim,ndim]))
    # tension 
    A[...,row,col] = A_v[...,inds_v.flatten()]
    # reverse factor 2 scaling
    if eng_conv:
        mask = ~np.eye(ndim,dtype=bool)
        A[...,mask] = 1/2 * A[...,mask]
    return A

def to_voigt(A: np.ndarray, 
             eng_conv: bool = False) -> np.ndarray:
    """
    Convert 2nd rank tensor into from the standard matrix represenation to its 
    Voigt representation.

    Parameters
    ----------
    A : np.ndarray
        2nd rank tensor in matrix represenation shape (...,ndim, ndim).
    eng_conv : bool 
        if True, engineering convention is applied meaning the shear components
        are scaled by a factor of two in Voigt notation. Usually applies only 
        to strains.

    Returns
    -------
    A_v : np.ndarray
        2nd rank tensor in Voigt notation shape (...,(ndim**2 + ndim) /2).
    """
    #
    ndim = A.shape[-1]
    nv = int((ndim**2 + ndim) /2)
    #
    row = np.array([0,1,2][:ndim]+[1,0,0][3-(nv-ndim):])
    col = np.array([0,1,2][:ndim]+[2,2,1][3-(nv-ndim):])
    A_v = A[...,row,col]
    # apply factor 2 scaling
    if eng_conv:
        A_v[...,ndim:] = 2 * A_v[...,ndim:]
    return A_v

def voigt_index(i: np.ndarray, 
                j: np.ndarray, 
                ndim: int) -> np.ndarray:
    """
    Map tensor index pairs (i,j) to Voigt indices for arbitrary dimension.
    Ordering is consistent with to_voigt.

    Parameters
    ----------
    i : int or np.ndarray
        first tensor index (0-based).
    j : int or np.ndarray
        second tensor index (0-based).
    ndim : int
        spatial dimension.

    Returns
    -------
    alpha : np.ndarray
        Voigt index/indices (0-based).
    """
    #
    nv = int((ndim**2 + ndim) /2)
    row, col = voigt_pair(np.arange(nv), ndim)
    # lookup table
    table = np.zeros((ndim,ndim), dtype=int)
    table[row,col] = np.arange(nv)
    table[col,row] = np.arange(nv)
    return table[np.asarray(i),np.asarray(j)]

def voigt_pair(alpha: np.ndarray, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Map Voigt indices back to tensor index pairs (i,j). Ordering is consistent
    with to_voigt.

    Parameters
    ----------
    alpha : int or np.ndarray
        Voigt index/indices (0-based).
    ndim : int
        spatial dimension.

    Returns
    -------
    i : np.ndarray
        first tensor index (0-based).
    j : np.ndarray
        second tensor index (0-based).
    """
    alpha = np.asarray(alpha)
    #
    nv = int((ndim**2 + ndim) /2)
    # same ordering as in to_voigt
    row = np.array([0,1,2][:ndim]+[1,0,0][3-(nv-ndim):])
    col = np.array([0,1,2][:ndim]+[2,2,1][3-(nv-ndim):])
    return row[alpha], col[alpha]