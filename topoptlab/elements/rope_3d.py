# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_rope_3d(E: float | np.ndarray = 1.,
               A: float | np.ndarray = 1.,
               l: float | np.ndarray = 1.,
               n: np.ndarray = np.array([1., 0., 0.]),
               **kwargs: Any) -> np.ndarray:
    """
    Create element matrices for the mixed (displacement-force) formulation
    of linear 2-node bar elements in 3D. The axial force N is the Lagrange
    multiplier enforcing compatibility. Ropes are obtained by bounding
    N >= 0, so the element itself is linear. Degrees of freedom are ordered
    (u1x, u1y, u1z, u2x, u2y, u2z, N), i. e. the multiplier is appended to
    the standard edofMat (see topoptlab.fem.append_multipliers).

    The element matrix reads

        [[ 0  , c        ],
         [ c^T, -l/(E*A) ]]

    with the compatibility operator c = [-n, n]. Eliminating N recovers the
    standard bar stiffness matrix.

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
    Ke : np.ndarray, shape (n_bars,7,7)
        element matrices.

    """
    n = np.atleast_2d(np.asarray(n, dtype=float))
    n = n / np.linalg.norm(n, axis=1)[:, None]
    f = np.atleast_1d(np.asarray(l, dtype=float)/(E*A))
    cx, cy, cz, f = np.broadcast_arrays(n[:, 0], n[:, 1], n[:, 2], f)
    z = np.zeros_like(f)
    return np.column_stack((z, z, z, z, z, z, -cx,
                            z, z, z, z, z, z, -cy,
                            z, z, z, z, z, z, -cz,
                            z, z, z, z, z, z, cx,
                            z, z, z, z, z, z, cy,
                            z, z, z, z, z, z, cz,
                            -cx, -cy, -cz, cx, cy, cz, -f)).reshape(-1, 7, 7)
