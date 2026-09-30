# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lf_bodyforce_2d(b: np.ndarray = np.array([0,-1]),
                    l: np.ndarray = np.array([1.,1.]),
                    t: float = 1.,
                    **kwargs: Any) -> np.ndarray:
    """
    Compute nodal forces on bilinear quadrilateral Lagrangian element
    (1st order) due to bodyforce (e. g. gravity) via analytical integration.
    Element is a parallelogram.

    Parameters
    ----------
    b : np.ndarray shape (2)
        body force (e. g. density*gravity_acceleration)
    l : np.ndarray (2)
        side length of element.
    t : float
        thickness of element.

    Returns
    -------
    fe : np.ndarray, shape (8,1)
        nodal forces.

    """
    A = l[0]*l[1] / 4
    return t*A*np.array([[b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]],
                         [b[0]],
                         [b[1]]])
