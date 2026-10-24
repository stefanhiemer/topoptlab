# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_inertia_1d(v: np.ndarray,
                  l: np.ndarray = np.array([1.]),
                  **kwargs: Any) -> np.ndarray:
    """
    Create element inertia (convective + Newton-linearization) matrix for
    the advection term v.grad(v) of the Navier-Stokes equations in 1D with
    linear interval Lagrangian elements.

    Parameters
    ----------
    v : np.ndarray (2)
        nodal velocities.
    l : np.ndarray (1)
        side length of element.

    Returns
    -------
    Ke : np.ndarray, shape (nels,2,2)
        element inertia matrix.

    """
    return np.column_stack((-2*v[0]/3 + v[1]/6, v[0]/6 + v[1]/3,
                            -v[0]/3 - v[1]/6, -v[0]/6 + 2*v[1]/3)).reshape(-1,2,2)

def lf_convective_1d(v: np.ndarray,
                     l: np.ndarray = np.array([1.]),
                     **kwargs: Any) -> np.ndarray:
    """
    Create element convective term v.grad(v) of the Navier-Stokes equations
    in 1D with linear interval Lagrangian elements, discretized at the
    nodal velocities v.

    Parameters
    ----------
    v : np.ndarray (2)
        nodal velocities.
    l : np.ndarray (1)
        side length of element.

    Returns
    -------
    fe : np.ndarray, shape (nels,2,1)
        element convective term.

    """
    return np.column_stack((-v[0]**2/3 + v[0]*v[1]/6 + v[1]**2/6,
                            -v[0]**2/6 - v[0]*v[1]/6 + v[1]**2/3)).reshape(-1,2,1)
