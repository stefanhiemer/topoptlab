# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_bar_1d(E: float | np.ndarray = 1.,
              A: float | np.ndarray = 1.,
              l: float | np.ndarray = 1.,
              **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrices for linear 2-node bar (truss) elements
    in 1D.

    Parameters
    ----------
    E : float or np.ndarray, shape (n_bars)
        Young's modulus.
    A : float or np.ndarray, shape (n_bars)
        cross-sectional area.
    l : float or np.ndarray, shape (n_bars)
        length of elements.

    Returns
    -------
    Ke : np.ndarray, shape (n_bars,2,2)
        element stiffness matrices.

    """
    k = np.atleast_1d(E*A/np.asarray(l, dtype=float))
    return np.column_stack((k, -k,
                            -k, k)).reshape(-1, 2, 2)
