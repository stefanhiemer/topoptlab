# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_rope_1d(E: float | np.ndarray = 1.,
               A: float | np.ndarray = 1.,
               l: float | np.ndarray = 1.,
               **kwargs: Any) -> np.ndarray:
    """
    Create element matrices for the mixed (displacement-force) formulation
    of linear 2-node bar elements in 1D. The axial force N is the Lagrange
    multiplier enforcing compatibility. Ropes are obtained by bounding
    N >= 0, so the element itself is linear. Degrees of freedom are ordered
    (u1, u2, N), i. e. the multiplier is appended to the standard edofMat
    (see topoptlab.fem.append_multipliers).

    The element matrix reads

        [[ 0  , c        ],
         [ c^T, -l/(E*A) ]]

    with the compatibility operator c = [-1, 1]. Eliminating N recovers the
    standard bar stiffness matrix.

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
    Ke : np.ndarray, shape (n_bars,3,3)
        element matrices.

    """
    f = np.atleast_1d(np.asarray(l, dtype=float)/(E*A))
    z = np.zeros_like(f)
    o = np.ones_like(f)
    return np.column_stack((z, z, -o,
                            z, z, o,
                            -o, o, -f)).reshape(-1, 3, 3)
