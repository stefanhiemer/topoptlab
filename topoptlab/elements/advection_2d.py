# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_advection_2d(v: np.ndarray,
                    l: np.ndarray = np.array([1.,1.]),
                    g: np.ndarray = np.array([0.]),
                    **kwargs: Any) -> np.ndarray:
    """
    Create element advection matrix for a scalar field in 2D with bilinear
    quadrilateral Lagrangian elements.

    Parameters
    ----------
    v : np.ndarray (8)
        nodal velocities.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element advection matrix.

    """
    return np.column_stack((-l[0]*v[1]/12 - l[0]*v[3]/36 - l[0]*v[5]/72 - l[0]*v[7]/24 - l[1]*v[0]/12 + l[1]*v[1]*np.tan(g[0])/12 - l[1]*v[2]/24 + l[1]*v[3]*np.tan(g[0])/24 - l[1]*v[4]/72 + l[1]*v[5]*np.tan(g[0])/72 - l[1]*v[6]/36 + l[1]*v[7]*np.tan(g[0])/36,
                             -l[0]*v[1]/36 - l[0]*v[3]/36 - l[0]*v[5]/72 - l[0]*v[7]/72 + l[1]*v[0]/12 - l[1]*v[1]*np.tan(g[0])/12 + l[1]*v[2]/24 - l[1]*v[3]*np.tan(g[0])/24 + l[1]*v[4]/72 - l[1]*v[5]*np.tan(g[0])/72 + l[1]*v[6]/36 - l[1]*v[7]*np.tan(g[0])/36,
                             l[0]*v[1]/36 + l[0]*v[3]/36 + l[0]*v[5]/72 + l[0]*v[7]/72 + l[1]*v[0]/36 - l[1]*v[1]*np.tan(g[0])/36 + l[1]*v[2]/72 - l[1]*v[3]*np.tan(g[0])/72 + l[1]*v[4]/72 - l[1]*v[5]*np.tan(g[0])/72 + l[1]*v[6]/36 - l[1]*v[7]*np.tan(g[0])/36,
                             l[0]*v[1]/12 + l[0]*v[3]/36 + l[0]*v[5]/72 + l[0]*v[7]/24 - l[1]*v[0]/36 + l[1]*v[1]*np.tan(g[0])/36 - l[1]*v[2]/72 + l[1]*v[3]*np.tan(g[0])/72 - l[1]*v[4]/72 + l[1]*v[5]*np.tan(g[0])/72 - l[1]*v[6]/36 + l[1]*v[7]*np.tan(g[0])/36,
                            -l[0]*v[1]/36 - l[0]*v[3]/36 - l[0]*v[5]/72 - l[0]*v[7]/72 - l[1]*v[0]/24 + l[1]*v[1]*np.tan(g[0])/24 - l[1]*v[2]/12 + l[1]*v[3]*np.tan(g[0])/12 - l[1]*v[4]/36 + l[1]*v[5]*np.tan(g[0])/36 - l[1]*v[6]/72 + l[1]*v[7]*np.tan(g[0])/72,
                             -l[0]*v[1]/36 - l[0]*v[3]/12 - l[0]*v[5]/24 - l[0]*v[7]/72 + l[1]*v[0]/24 - l[1]*v[1]*np.tan(g[0])/24 + l[1]*v[2]/12 - l[1]*v[3]*np.tan(g[0])/12 + l[1]*v[4]/36 - l[1]*v[5]*np.tan(g[0])/36 + l[1]*v[6]/72 - l[1]*v[7]*np.tan(g[0])/72,
                             l[0]*v[1]/36 + l[0]*v[3]/12 + l[0]*v[5]/24 + l[0]*v[7]/72 + l[1]*v[0]/72 - l[1]*v[1]*np.tan(g[0])/72 + l[1]*v[2]/36 - l[1]*v[3]*np.tan(g[0])/36 + l[1]*v[4]/36 - l[1]*v[5]*np.tan(g[0])/36 + l[1]*v[6]/72 - l[1]*v[7]*np.tan(g[0])/72,
                             l[0]*v[1]/36 + l[0]*v[3]/36 + l[0]*v[5]/72 + l[0]*v[7]/72 - l[1]*v[0]/72 + l[1]*v[1]*np.tan(g[0])/72 - l[1]*v[2]/36 + l[1]*v[3]*np.tan(g[0])/36 - l[1]*v[4]/36 + l[1]*v[5]*np.tan(g[0])/36 - l[1]*v[6]/72 + l[1]*v[7]*np.tan(g[0])/72,
                            -l[0]*v[1]/72 - l[0]*v[3]/72 - l[0]*v[5]/36 - l[0]*v[7]/36 - l[1]*v[0]/72 + l[1]*v[1]*np.tan(g[0])/72 - l[1]*v[2]/36 + l[1]*v[3]*np.tan(g[0])/36 - l[1]*v[4]/36 + l[1]*v[5]*np.tan(g[0])/36 - l[1]*v[6]/72 + l[1]*v[7]*np.tan(g[0])/72,
                             -l[0]*v[1]/72 - l[0]*v[3]/24 - l[0]*v[5]/12 - l[0]*v[7]/36 + l[1]*v[0]/72 - l[1]*v[1]*np.tan(g[0])/72 + l[1]*v[2]/36 - l[1]*v[3]*np.tan(g[0])/36 + l[1]*v[4]/36 - l[1]*v[5]*np.tan(g[0])/36 + l[1]*v[6]/72 - l[1]*v[7]*np.tan(g[0])/72,
                             l[0]*v[1]/72 + l[0]*v[3]/24 + l[0]*v[5]/12 + l[0]*v[7]/36 + l[1]*v[0]/72 - l[1]*v[1]*np.tan(g[0])/72 + l[1]*v[2]/36 - l[1]*v[3]*np.tan(g[0])/36 + l[1]*v[4]/12 - l[1]*v[5]*np.tan(g[0])/12 + l[1]*v[6]/24 - l[1]*v[7]*np.tan(g[0])/24,
                             l[0]*v[1]/72 + l[0]*v[3]/72 + l[0]*v[5]/36 + l[0]*v[7]/36 - l[1]*v[0]/72 + l[1]*v[1]*np.tan(g[0])/72 - l[1]*v[2]/36 + l[1]*v[3]*np.tan(g[0])/36 - l[1]*v[4]/12 + l[1]*v[5]*np.tan(g[0])/12 - l[1]*v[6]/24 + l[1]*v[7]*np.tan(g[0])/24,
                            -l[0]*v[1]/24 - l[0]*v[3]/72 - l[0]*v[5]/36 - l[0]*v[7]/12 - l[1]*v[0]/36 + l[1]*v[1]*np.tan(g[0])/36 - l[1]*v[2]/72 + l[1]*v[3]*np.tan(g[0])/72 - l[1]*v[4]/72 + l[1]*v[5]*np.tan(g[0])/72 - l[1]*v[6]/36 + l[1]*v[7]*np.tan(g[0])/36,
                             -l[0]*v[1]/72 - l[0]*v[3]/72 - l[0]*v[5]/36 - l[0]*v[7]/36 + l[1]*v[0]/36 - l[1]*v[1]*np.tan(g[0])/36 + l[1]*v[2]/72 - l[1]*v[3]*np.tan(g[0])/72 + l[1]*v[4]/72 - l[1]*v[5]*np.tan(g[0])/72 + l[1]*v[6]/36 - l[1]*v[7]*np.tan(g[0])/36,
                             l[0]*v[1]/72 + l[0]*v[3]/72 + l[0]*v[5]/36 + l[0]*v[7]/36 + l[1]*v[0]/36 - l[1]*v[1]*np.tan(g[0])/36 + l[1]*v[2]/72 - l[1]*v[3]*np.tan(g[0])/72 + l[1]*v[4]/24 - l[1]*v[5]*np.tan(g[0])/24 + l[1]*v[6]/12 - l[1]*v[7]*np.tan(g[0])/12,
                             l[0]*v[1]/24 + l[0]*v[3]/72 + l[0]*v[5]/36 + l[0]*v[7]/12 - l[1]*v[0]/36 + l[1]*v[1]*np.tan(g[0])/36 - l[1]*v[2]/72 + l[1]*v[3]*np.tan(g[0])/72 - l[1]*v[4]/24 + l[1]*v[5]*np.tan(g[0])/24 - l[1]*v[6]/12 + l[1]*v[7]*np.tan(g[0])/12)).reshape(4,4)

