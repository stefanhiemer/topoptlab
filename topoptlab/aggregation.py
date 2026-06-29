# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, List, Union
from warnings import warn

import numpy as np

def softmax_pnorm(values: np.ndarray,
                  p: float, 
                  axis: Union[None,int,List] = None,
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
                     p: float,
                     axis: Union[None, int, List] = None,
                     **kwargs: Any) -> np.ndarray:
    """
    First derivative of aggregation of values to single value via p-norm
    which is a soft maximum operator.

    Parameters
    ----------
    values : np.ndarray
        values to be aggregated of shape (n) or (n,c).
    p : float
        order of p-norm.
    axis : None or int or list
        axis over which to aggregate. If None, all values are aggregated.

    Returns
    -------
    aggreg_dx : np.ndarray
        first derivative of aggregated function.

    """
    #
    psum = (values**p).sum(axis=axis, keepdims=True)
    #
    factor = np.zeros_like(psum)
    mask = ~np.isclose(psum, 0.)
    #
    factor[mask] = psum[mask]**((1-p)/p)
    return factor * values**(p-1)

def softmax_ks(values: np.ndarray,
               exponent: float, 
               axis: Union[None,int,List] = None,
               **kwargs: Any) -> Union[float,np.ndarray]:
    """
    Aggregation of values to single value via the Kreisselmeier and Steinhauser function

    Kreisselmeier, Gerhard, and Reinhold Steinhauser. "Application of 
    vector performance optimization to a robust control loop design 
    for a fighter aircraft." International Journal of Control 37.2 (1983): 251-284.


    which is a soft maximum operator

    Parameters
    ----------
    values : np.ndarray
        values to be aggregated of shape (n) or (n,c).
    exponent : float
        KS exponent.
    axis : None or int or list
        axis over which to aggregate. If None, all values are aggregated.

    Returns
    -------
    aggreg : float or np.ndarray
        aggregated function.

    """
    
    return np.log((np.exp(exponent*values)).mean(axis=axis))/exponent

def softmax_ks_dx(values: np.ndarray, 
                  exponent: float,
                  axis: Union[None, int, List] = None,
                  **kwargs: Any) -> np.ndarray:
    """
    First derivative of aggregation of values to single value via 
    the Kreisselmeier and Steinhauser function

    Kreisselmeier, Gerhard, and Reinhold Steinhauser. "Application of 
    vector performance optimization to a robust control loop design 
    for a fighter aircraft." International Journal of Control 37.2 (1983): 251-284.


    which is a soft maximum operator.

    Parameters
    ----------
    values : np.ndarray
        values to be aggregated of shape (n) or (n,c).
    exponent : float
        KS exponent.
    axis : None or int or list
        axis over which to aggregate. If None, all values are aggregated.

    Returns
    -------
    aggreg_dx : np.ndarray
        first derivative of aggregated function.

    """
    exponentials = np.exp(exponent * values)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)

def interlaced_aggregation(values: np.ndarray,
                           n_regions: int) -> List[np.ndarray]:
    """
    Aggregate into interlaced regional groups as done in chapter 3 of

    Le, Chau, et al. "Stress-based topology optimization for continua." 
    Structural and Multidisciplinary Optimization 41.4 (2010): 605-620.
    

    """
    #
    inds = np.argsort(vals,axis=0)
    #
    return [inds[k::n_regions] for k in range(n_regions)]

def quantile_division(values: np.ndarray,
                      k: float, 
                      axis: Union[None,int,List],
                      **kwargs: Any) -> Union[float,np.ndarray]:
    """
    Divide values to k equal pieces based on sorting. Can be used to reduce element-wise
    constraints to k constraints.

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
    #
    inds = np.split(np.argsort(values), 
                    indices_or_sections=k, 
                    axis=0)
    #
    values[inds]
    return 