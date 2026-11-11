# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_bar_2d(E: float | np.ndarray = 1.,
              A: float | np.ndarray = 1.,
              l: float | np.ndarray = 1.,
              n: np.ndarray = np.array([1., 0.]),
              **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrices for linear 2-node bar (truss) elements
    in 2D. Degrees of freedom are ordered (u1x, u1y, u2x, u2y).

    Parameters
    ----------
    E : float or np.ndarray, shape (n_bars)
        Young's modulus.
    A : float or np.ndarray, shape (n_bars)
        cross-sectional area.
    l : float or np.ndarray, shape (n_bars)
        length of elements.
    n : np.ndarray, shape (2) or (n_bars,2)
        direction of bars pointing from node 1 to node 2. Is normalized
        internally.

    Returns
    -------
    Ke : np.ndarray, shape (n_bars,4,4)
        element stiffness matrices.

    """
    n = np.atleast_2d(np.asarray(n, dtype=float))
    n = n / np.linalg.norm(n, axis=1)[:, None]
    c, s = n[:, 0], n[:, 1]
    k = np.atleast_1d(E*A/np.asarray(l, dtype=float))[:, None, None]
    return k*np.column_stack((c*c, c*s, -c*c, -c*s,
                              c*s, s*s, -c*s, -s*s,
                              -c*c, -c*s, c*c, c*s,
                              -c*s, -s*s, c*s, s*s)).reshape(-1, 4, 4)
