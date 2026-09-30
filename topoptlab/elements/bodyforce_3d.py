# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lf_bodyforce_3d(b: np.ndarray = np.array([0.,-1.,0.]),
                    l: np.ndarray = np.array([1.,1.,1.]),
                    **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on trilinear hexahedral Lagrangian element (1st order)
    due to bodyforce (e. g. gravity) via analytical integration. Element is a
    parallelepiped.

    Parameters
    ----------
    b : np.ndarray shape (3)
        body force
    l : np.ndarray (3)
        side length of element

    Returns
    -------
    fe : np.ndarray, shape (24,1)
        nodal forces.

    """
    v = l[0]*l[1]*l[2] / 8
    return v*np.array([[b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]],
                       [b[0]],
                       [b[1]],
                       [b[2]]])
