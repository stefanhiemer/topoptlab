# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Union
#
import numpy as np

def update_history(xhist: List, 
                   x: np.ndarray,
                   xPhys_hist: Union[None, List], 
                   xPhys: np.ndarray,
                   obj_hist: Union[None, List], 
                   obj: float,
                   constrs_hist: Union[None, List], 
                   constrs: np.ndarray,
                   max_history: int) -> None:
    """
    Append current iterate to history lists and prune to max_history+1 entries.

    Parameters
    ----------
    xhist : list of np.ndarray
        history of design iterates.
    x : np.ndarray
        current design iterate.
    xPhys_hist : None or list of np.ndarray
        history of physical densities, or None if not tracked.
    xPhys : np.ndarray
        current physical densities.
    obj_hist : None or list of float
        history of objective values, or None if not tracked.
    obj : float
        current objective value.
    constrs_hist : None or list of np.ndarray
        history of constraint vectors, or None if not tracked.
    constrs : np.ndarray
        current constraint vector.
    max_history : int
        maximum number of iterates to retain.

    Returns
    -------
    None
    """
    # append history
    xhist.append(x.copy())
    if xPhys_hist is not None:
        xPhys_hist.append(xPhys.copy())
    if obj_hist is not None:
        obj_hist.append(obj)
    if constrs_hist is not None:
        constrs_hist.append(constrs.copy())
    # prune history
    if len(xhist) > max_history+1:
        del xhist[:len(xhist)-max_history-1]
    if xPhys_hist is not None and len(xPhys_hist) > max_history+1:
        del xPhys_hist[:len(xPhys_hist)-max_history-1]
    if obj_hist is not None and len(obj_hist) > max_history+1:
        del obj_hist[:len(obj_hist)-max_history-1]
    if constrs_hist is not None and len(constrs_hist) > max_history+1:
        del constrs_hist[:len(constrs_hist)-max_history-1]
    return