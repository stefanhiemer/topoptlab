# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_bar_3d(E: float | np.ndarray = 1.,
              A: float | np.ndarray = 1.,
              l: float | np.ndarray = 1.,
              n: np.ndarray = np.array([1., 0., 0.]),
              **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrices for linear 2-node bar (truss) elements
    in 3D. Degrees of freedom are ordered (u1x, u1y, u1z, u2x, u2y, u2z).

    Parameters
    ----------
    E : float or np.ndarray, shape (n_bars)
        Young's modulus.
    A : float or np.ndarray, shape (n_bars)
        cross-sectional area.
    l : float or np.ndarray, shape (n_bars)
        length of elements.
    n : np.ndarray, shape (3) or (n_bars,3)
        direction of bars pointing from node 1 to node 2. Is normalized
        internally.

    Returns
    -------
    Ke : np.ndarray, shape (n_bars,6,6)
        element stiffness matrices.

    """
    n = np.atleast_2d(np.asarray(n, dtype=float))
    n = n / np.linalg.norm(n, axis=1)[:, None]
    cx, cy, cz = n[:, 0], n[:, 1], n[:, 2]
    k = np.atleast_1d(E*A/np.asarray(l, dtype=float))[:, None, None]
    return k*np.column_stack((cx*cx, cx*cy, cx*cz, -cx*cx, -cx*cy, -cx*cz,
                              cx*cy, cy*cy, cy*cz, -cx*cy, -cy*cy, -cy*cz,
                              cx*cz, cy*cz, cz*cz, -cx*cz, -cy*cz, -cz*cz,
                              -cx*cx, -cx*cy, -cx*cz, cx*cx, cx*cy, cx*cz,
                              -cx*cy, -cy*cy, -cy*cz, cx*cy, cy*cy, cy*cz,
                              -cx*cz, -cy*cz, -cz*cz, cx*cz, cy*cz, cz*cz)
                             ).reshape(-1, 6, 6)
