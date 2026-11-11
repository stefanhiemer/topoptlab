# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def bar_geometry(coords: np.ndarray,
                 bars: np.ndarray,
                 **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute lengths and unit directions of bar (truss) elements from nodal
    coordinates and bar connectivity.

    Parameters
    ----------
    coords : np.ndarray, shape (n_nodes,ndim)
        nodal coordinates.
    bars : np.ndarray, shape (n_bars,2)
        node indices of each bar. Direction points from bars[:,0] to
        bars[:,1].

    Returns
    -------
    l : np.ndarray, shape (n_bars)
        length of each bar.
    n : np.ndarray, shape (n_bars,ndim)
        unit direction of each bar.

    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords[:, None]
    bars = np.asarray(bars)
    #
    d = coords[bars[:, 1]] - coords[bars[:, 0]]
    l = np.linalg.norm(d, axis=1)
    if np.any(l == 0.):
        raise ValueError("Bars with zero length: ",
                         np.flatnonzero(l == 0.))
    return l, d / l[:, None]

def create_edofMat(bars: np.ndarray,
                   nnode_dof: int,
                   dtype: type = np.int32,
                   **kwargs: Any) -> np.ndarray:
    """
    Create element degree of freedom matrix for bar (truss) elements with
    arbitrary connectivity. Nodal dofs are ordered node-wise, i. e. dof k of
    node i is i*nnode_dof + k.

    Parameters
    ----------
    bars : np.ndarray, shape (n_bars,2)
        node indices of each bar.
    nnode_dof : int
        number of node degrees of freedom (usually ndim).

    Returns
    -------
    edofMat : np.ndarray, shape (n_bars,2*nnode_dof)
        element degree of freedom matrix.

    """
    bars = np.asarray(bars, dtype=dtype)
    edofMat = np.repeat(bars * nnode_dof, nnode_dof, axis=1)
    edofMat = edofMat + np.tile(np.arange(nnode_dof, dtype=dtype), 2)[None, :]
    return edofMat
