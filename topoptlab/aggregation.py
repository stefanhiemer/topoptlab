# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, List, Union
from warnings import warn

import numpy as np

def softmax_pnorm(values: np.ndarray,
                  p: float, 
                  axis: Union[None,int,List],
                  **kwargs: Any) -> Union[float,np.ndarray]:
    """
    Aggregation of values to single value via p-norm which is a 
    soft maximum operator.

    Parameters
    ----------
    values : np.ndarray
        values to be aggregated of shape (n) or (n,c).
    p : float
        order of p-norm
    axis : None or int or list
        axis over which to aggregate. If None, all values are aggregated.

    Returns
    -------
    aggreg : float or np.ndarray
        aggregated function.

    """
    
    return (values**p).sum(axis=axis)**(1/p)

def softmax_pnorm_dx(values: np.ndarray,
                     values_dx: np.ndarray,
                     p: float,
                     **kwargs: Any) -> np.ndarray:
    """
    First derivative of aggregation of values to single value via p-norm 
    which is a soft maximum operator.

    Parameters
    ----------
    values : np.ndarray
        values to be aggregated of shape (n) or (n,c).
    values_dx: np.ndarray,
        first derivative with regards to x.
    p : float
        order of p-norm
    axis : None or int or list
        axis over which to aggregate. If None, all values are aggregated.

    Returns
    -------
    aggreg_dx : np.ndarray
        first derivative of aggregated function.

    """


    return (values**p).sum(axis=axis)**((1-p)/p) * values**(p-1) * values_dx