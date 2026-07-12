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

def orthotropic(k: Union[None,np.ndarray] = None,
                kx: Union[None,float] = None,
                ky: Union[None,float] = None,
                kz: Union[None,float] = None) -> np.ndarray:
    """
    Orthotropic (diagonal) rank-2 material property tensor, e.g. heat
    conductivity or thermal expansion, built from one value per axis.

    Can be called in two ways:

    - ``orthotropic(k=array)`` — pass a 1-D array of per-axis values directly.
    - ``orthotropic(kx=..., ky=..., kz=...)`` — pass individual components;
      ``kz`` may be omitted for 2-D problems.

    Parameters
    ----------
    k : np.ndarray, shape (ndim,) or None
        Per-axis property values as a 1-D array.  If given, ``kx``/``ky``/``kz``
        are ignored.
    kx, ky, kz : float or None
        Individual per-axis components.  Used when ``k`` is None.
        ``kz`` may be omitted for 2-D problems.

    Returns
    -------
    K : np.ndarray, shape (ndim, ndim)
        Diagonal property tensor.
    """
    if k is not None:
        return np.diag(k)
    else:
        return np.diag([_k for _k in (kx, ky, kz) if _k is not None])