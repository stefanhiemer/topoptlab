# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial
from cProfile import Profile
from datetime import datetime
import inspect
#
import numpy as np

def initialize_design(n: int,
                      initial_guess: Union[None, Dict[str, np.ndarray]],
                      volfrac: Union[None, float],
                      n_mat: int = 1,
                      n_x_channels: Union[None, int] = None,
                      **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize design variables x and physical densities xPhys.

    Parameters
    ----------
    n : int
        Total number of design elements.
    initial_guess : dict or None
        Optional initial values.  Recognised keys:

        ``"x"``
            Initial design variables, shape (n, n_mat).  Defaults to uniform
            ``volfrac`` (or 0.5 if ``volfrac`` is None).
        ``"xPhys"``
            Initial physical densities, shape (n, n_mat).  Defaults to a copy
            of ``x``.

    volfrac : float or None
        Volume fraction used to fill ``x`` when no initial guess is given.
        If None, defaults to 0.5.
    n_mat : int
        Number of materials; determines the second axis of x and xPhys.

    Returns
    -------
    x : np.ndarray, shape (n, n_mat)
        Design variables.
    xPhys : np.ndarray, shape (n, n_mat)
        Physical densities.
    """
    if n_x_channels is None:
        n_x_channels = n_mat
    if initial_guess is None or \
       "x" not in initial_guess:
        #
        if volfrac is None:
            fill_val = 0.5
        elif isinstance(volfrac,float) or \
             (isinstance(volfrac,np.ndarray) and \
              len(volfrac) == 0) or \
             (isinstance(volfrac,np.ndarray) and \
              len(volfrac) == 1):
            fill_val = np.squeeze(volfrac)
        elif isinstance(volfrac,np.ndarray):
            fill_val = volfrac/volfrac.shape[0]
        else:
            raise TypeError("volfrac must be of type float or np.ndarray. Current type: ", 
                            type(volfrac))
        #
        x = np.full(shape=(n, n_x_channels),
                    fill_value=fill_val,
                    dtype=float,
                    order='F')
    else:
        x = initial_guess["x"]
    #
    if initial_guess is None or "xPhys" not in initial_guess:
        if n_x_channels != n_mat:
            xPhys = np.full(shape=(n, n_mat),
                            fill_value=fill_val,
                            dtype=float,
                            order='F')
        else:
            xPhys = x.copy()
    else:
        xPhys = initial_guess["xPhys"]
    return x, xPhys