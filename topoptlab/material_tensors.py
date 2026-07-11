# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Dict,List,Tuple,Union

import numpy as np

def isotropic(ndim: int,
              k: float = 1.) -> np.ndarray:
    """
    Isotropic rank-2 material property tensor, e.g. heat conductivity or
    thermal expansion, as ``k * I``.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    k : float
        isotropic (scalar) value of the property.

    Returns
    -------
    K : np.ndarray, shape (ndim, ndim)
        property tensor.
    """
    return np.eye(ndim)*k

def orthotropic(k: np.ndarray) -> np.ndarray:
    """
    Orthotropic (diagonal) rank-2 material property tensor, e.g. heat
    conductivity or thermal expansion, built from one value per axis.

    Parameters
    ----------
    k : np.ndarray, shape (ndim,)
        property value along each axis.

    Returns
    -------
    K : np.ndarray, shape (ndim, ndim)
        property tensor.
    """
    return np.diag(k)