# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_advection_1d(v: np.ndarray,
                    l: np.ndarray = np.array([1.]),
                    **kwargs: Any) -> np.ndarray:
    """
    Create element advection matrix for a scalar field in 1D with linear
    interval Lagrangian elements.

    Parameters
    ----------
    v : np.ndarray (2)
        nodal velocities.
    l : np.ndarray (1)
        side length of element.

    Returns
    -------
    Ke : np.ndarray, shape (2,2)
        element advection matrix.

    """
    return np.column_stack((-v[0]/3 - v[1]/6, v[0]/3 + v[1]/6,
                            -v[0]/6 - v[1]/3, v[0]/6 + v[1]/3)).reshape(2,2)
