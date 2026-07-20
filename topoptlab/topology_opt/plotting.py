# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial
from cProfile import Profile
from datetime import datetime
import inspect
#
import numpy as np
#
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def initialize_plotting(xPhys: np.ndarray,
                        mapping: Callable,
                        ndim: int,
                        img_kw: Dict = {"cmap": 'gray',
                                        "interpolation": 'none',
                                        "norm": Normalize(vmin=-1, vmax=0)},
                        **kwargs: Any) -> Callable:
    """
    Set up an interactive matplotlib figure for live density-field updates.

    Parameters
    ----------
    xPhys : np.ndarray
        Initial physical density field; passed through ``mapping`` to produce
        the displayed image.
    mapping : callable
        Maps the element-indexed density array to a 2-D image array
        (e.g. ``map_eltoimg``).
    img_kw : dict
        Keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    plotfunc : callable
        ``im.set_array`` bound to the created ``AxesImage``; call it each
        iteration with the updated mapped array to refresh the display.
    """
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    if ndim == 2:
        fig,ax = plt.subplots(1,1)
        im = ax.imshow(mapping(-xPhys), 
                       **img_kw)
        plotfunc = im.set_array
    elif ndim == 3:
        raise NotImplementedError("Plotting in 3D not implemented.")
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    ax.axis("off")
    fig.show()
    return fig, plotfunc