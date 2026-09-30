# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lm_mass_3d(p: float = 1.0,
               l: np.ndarray = np.array([1.,1.,1.])):
    """
    Create element mass matrix for scalar field in 3D with trilinear hexahedral
    elements.

    Parameters
    ----------
    p : float
        density of element
    l : np.ndarray (3)
        side length of element

    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.

    """
    v = l[0]*l[1]*l[2]
    return p*v*np.array([[1/27, 1/54, 1/108, 1/54, 1/54, 1/108, 1/216, 1/108],
                         [1/54, 1/27, 1/54, 1/108, 1/108, 1/54, 1/108, 1/216],
                         [1/108, 1/54, 1/27, 1/54, 1/216, 1/108, 1/54, 1/108],
                         [1/54, 1/108, 1/54, 1/27, 1/108, 1/216, 1/108, 1/54],
                         [1/54, 1/108, 1/216, 1/108, 1/27, 1/54, 1/108, 1/54],
                         [1/108, 1/54, 1/108, 1/216, 1/54, 1/27, 1/54, 1/108],
                         [1/216, 1/108, 1/54, 1/108, 1/108, 1/54, 1/27, 1/54],
                         [1/108, 1/216, 1/108, 1/54, 1/54, 1/108, 1/54, 1/27]])
