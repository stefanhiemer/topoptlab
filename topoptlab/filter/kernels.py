# SPDX-License-Identifier: GPL-3.0-or-later
from typing import  Union

import numpy as np

def hat_kernel(x: np.ndarray, 
               y: np.ndarray, 
               rmin: float, 
               z: Union[None,np.ndarray]) -> np.ndarray:
    """
    Hat (cone) kernel: linear decay from rmin to zero at the cutoff radius.

    Computes the Euclidean distance r from coordinate offset arrays and
    returns max(0, rmin - r).

    Parameters
    ----------
    x : np.ndarray
        offset coordinates in x direction.
    y : np.ndarray
        offset coordinates in y direction.
    rmin : float
        cutoff radius in element widths.
    z : np.ndarray or None
        offset coordinates in z direction. If None, a 2D distance is used.

    Returns
    -------
    kernel : np.ndarray
        kernel weights of the same shape as the input arrays.
    """
    if z is None:
        r = np.sqrt(x**2 + y**2)
    else:
        r = np.sqrt(x**2 + y**2 + z**2)
    return np.maximum(0., rmin - r)
