# SPDX-License-Identifier: GPL-3.0-or-later
import inspect
from typing import Dict, List, Tuple, Union
#
import numpy as np

def initialize_history(max_history: int,
                       accelerator_kw: Dict,
                       continuation_kw: Union[None, Dict],
                       convergence_kw: Dict,
                       x: np.ndarray,
                       xPhys: np.ndarray,
                       constrs: np.ndarray) -> Tuple[int, Union[None, Dict], Dict]:
    """
    Allocate the iteration history dict and finalize ``max_history``.

    Also seeds ``continuation_kw["stop_flag"]`` if continuation is active.

    Returns
    -------
    max_history : int
        Effective history length after accounting for the accelerator.
    continuation_kw : dict or None
        Input dict extended with ``"stop_flag"``, or None unchanged.
    hist : dict
        ``{"xhist", "xPhys_hist", "obj_hist", "constrs_hist"}`` pre-filled
        with ``max_history`` copies of the initial iterates.
    """
    max_history = int(np.maximum(max_history, accelerator_kw.get("max_history", 0)))
    _cont_params = [inspect.signature(f).parameters
                    for f in (continuation_kw["funcs"]
                              if continuation_kw is not None else [])]
    _need_xPhys_hist = any("xPhys_hist" in p for p in _cont_params)
    if not _need_xPhys_hist and "mode" in convergence_kw.keys():
        _need_xPhys_hist = convergence_kw["mode"] == "xPhys"
    if continuation_kw is not None:
        continuation_kw["stop_flag"] = [False] * len(continuation_kw["funcs"])
    hist = {"xhist":        [x.copy() for _ in np.arange(max_history)],
            "xPhys_hist":   [xPhys.copy() for _ in np.arange(max_history)]
                            if _need_xPhys_hist else None,
            "obj_hist":     [0. for _ in np.arange(max_history)],
            "constrs_hist": [constrs.copy() for _ in np.arange(max_history)]}
    return max_history, continuation_kw, hist


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