# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lm_mass_2d(p: float = 1.,
               l: np.ndarray = np.array([1.,1.]),
               t: float = 1.,
               **kwargs: Any) -> np.ndarray:
    """
    Create mass matrix for vector field in 2D with bilinear quadrilateral
    Lagrangian elements.

    Parameters
    ----------
    p : float
        density of element
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.

    """
    v = l[0]*l[1]*t
    return p*v*np.array([[1/9, 0, 1/18, 0, 1/36, 0, 1/18, 0],
                         [0, 1/9, 0, 1/18, 0, 1/36, 0, 1/18],
                         [1/18, 0, 1/9, 0, 1/18, 0, 1/36, 0],
                         [0, 1/18, 0, 1/9, 0, 1/18, 0, 1/36],
                         [1/36, 0, 1/18, 0, 1/9, 0, 1/18, 0],
                         [0, 1/36, 0, 1/18, 0, 1/9, 0, 1/18],
                         [1/18, 0, 1/36, 0, 1/18, 0, 1/9, 0],
                         [0, 1/18, 0, 1/36, 0, 1/18, 0, 1/9]])
