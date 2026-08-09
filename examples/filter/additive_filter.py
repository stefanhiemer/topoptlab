# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.utils import map_eltoimg, map_imgtoel
from topoptlab.output_designs import threshold
from topoptlab.filter.amfilter_langelaar import LangelaarFilter


def display(x: np.ndarray,
            nelx: int,
            nely: int,
            baseplates: List[str] = ["S","N","W","E"]):
    """
    Visualise additive-manufacturing overhang violations for multiple build
    directions.

    Applies the Langelaar AM filter for each baseplate orientation and renders
    one subplot per direction.  Solid elements that survive the filter are
    shown in black; solid elements that are flagged as unsupported overhangs
    (filtered value < 0.4) are highlighted in red.  Void elements remain white.

    The top-left panel shows the original thresholded design for reference;
    the top-right panel is left blank.  The four remaining panels (rows 1-2)
    correspond to the baseplates in the order given by ``baseplates``.

    Parameters
    ----------
    x : np.ndarray, shape (n_el, 1)
        Binary element densities (0 = void, 1 = solid).
    nelx : int
        Number of elements in the x direction.
    nely : int
        Number of elements in the y direction.
    baseplates : list of str
        Build-plate orientations to evaluate.  Each entry is passed to
        ``LangelaarFilter`` as ``baseplate``.  Supported values are
        ``'N'``, ``'S'``, ``'E'``, ``'W'``.
    """
    #
    mapping = partial(map_eltoimg,  nelx=nelx, nely=nely)
    invmapping = partial(map_imgtoel, nelx=nelx, nely=nely)
    #
    x_img = mapping(x)
    #
    fig, axs = plt.subplots(3, 2)
    #
    axs[0,0].imshow(-x_img,
                    cmap="gray",
                    interpolation='none',
                    norm=Normalize(vmin=-1, vmax=0))
    axs[0,0].tick_params(axis='both', 
                         which='both',
                         bottom=False, left=False,
                         labelbottom=False, labelleft=False)
    axs[0,1].tick_params(axis='both', 
                         which='both',
                         bottom=False, left=False,
                         labelbottom=False, labelleft=False)
    #
    for i,baseplate in enumerate(baseplates):
        #
        row,col = int(1 + np.floor(i/2)),i%2
        #
        x_filt = LangelaarFilter(nelx=nelx,
                                 nely=nely,
                                 n_constr=0,
                                 mapping=mapping,
                                 invmapping=invmapping,
                                 baseplate=baseplate).apply_filter(x)
        #
        x_filt_img = mapping(x_filt)
        img = np.ones(x_img.shape[:-1] + (3,))
        img[np.isclose(x_img[..., 0],1)] = [0, 0, 0]
        img[np.isclose(x_img[..., 0],1.) & (x_filt_img[...,0] < 0.4),0] = 1
        axs[row,col].imshow(img,
                            cmap='viridis', 
                            interpolation='none',
                            norm=Normalize(vmin=0, vmax=1))
        axs[row,col].set_title(f"Baseplate {baseplate}")
        axs[row,col].tick_params(axis='both', 
                                 which='both',
                                 bottom=False, left=False,
                                 labelbottom=False, labelleft=False)
    plt.show()
    return

if __name__ == "__main__":
    nelx = 240
    nely = 80
    nelz = None
    #
    x = np.loadtxt("mbb2d_240x80_v0.5_ft0.csv", delimiter=",")[:, None]
    x = threshold(xPhys=x, volfrac=0.5)
    #
    display(x=x, nelx=nelx, nely=nely)
    #
    x = np.array([0,1,1,
                  0,1,0,
                  1,1,0])[:,None]
    #
    display(x=x, nelx=3, nely=3)
