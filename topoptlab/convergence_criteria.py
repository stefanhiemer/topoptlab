# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,List,Union

import numpy as np

def max_design_change(xhist: List, 
                      xPhys_hist: Union[None,List[np.ndarray]],
                      obj_hist: Union[None,List[float]],
                      constrs_hist: Union[None,List[np.ndarray]],
                      mode: str = "x", 
                      **kwargs: Any) -> float:
    """
    Maximum change in design or physical variables between the last two
    iterates (inf-norm).

    Parameters
    ----------
    xhist : list of np.ndarray
        history of design iterates. Must contain at least two entries;
        the last element is x_[k] and the second-to-last is x_[k-1].
    xPhys_hist : None or list of np.ndarray
        history of physical density iterates, same convention as xhist.
        Required when mode is "xPhys", ignored otherwise.
    obj_hist : None or list of float
        history of objective values (not used by this function, kept for
        a consistent convergence-function signature).
    constrs_hist : None or list of np.ndarray
        history of constraint vectors (not used by this function, kept for
        a consistent convergence-function signature).
    mode : str
        which variable to measure the change in. ``"x"`` uses the design
        variables (default); ``"xPhys"`` uses the physical densities.

    Returns
    -------
    change : float
        inf-norm of the difference between the last two iterates.
    """
    if mode == "x":
        return np.abs(xhist[-1] - xhist[-2]).max()
    elif mode == "xPhys":
        return np.abs(xPhys_hist[-1] - xPhys_hist[-2]).max()
    else:
        raise ValueError("unknown mode.")


def norm_design_change(xhist: List,
                       xPhys_hist: Union[None, List[np.ndarray]],
                       obj_hist: Union[None, List[float]],
                       constrs_hist: Union[None, List[np.ndarray]],
                       mode: str = "x",
                       ord: Union[None, int, float, str] = None,
                       **kwargs: Any) -> float:
    """
    Norm of the change in design or physical variables between the last two
    iterates.

    Parameters
    ----------
    xhist : list of np.ndarray
        history of design iterates. Must contain at least two entries;
        the last element is x_[k] and the second-to-last is x_[k-1].
    xPhys_hist : None or list of np.ndarray
        history of physical density iterates, same convention as xhist.
        Required when mode is "xPhys", ignored otherwise.
    obj_hist : None or list of float
        history of objective values (not used by this function, kept for
        a consistent convergence-function signature).
    constrs_hist : None or list of np.ndarray
        history of constraint vectors (not used by this function, kept for
        a consistent convergence-function signature).
    mode : str
        which variable to measure the change in. ``"x"`` uses the design
        variables (default); ``"xPhys"`` uses the physical densities.
    ord : None, int, float, or str
        order of the norm passed to ``np.linalg.norm``. None gives the
        2-norm; ``np.inf`` gives the inf-norm. See ``np.linalg.norm`` for
        all options.

    Returns
    -------
    change : float
        norm of the difference between the last two iterates.
    """
    if mode == "x":
        return np.linalg.norm(xhist[-1] - xhist[-2], ord=ord)
    elif mode == "xPhys":
        return np.linalg.norm(xPhys_hist[-1] - xPhys_hist[-2], ord=ord)
    else:
        raise ValueError("unknown mode.")
