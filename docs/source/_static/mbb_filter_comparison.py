# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate the MBB beam designs (60x20, rmin=2.4) used to illustrate the effect
of filtering:

1. no filter (checkerboard)
2. sensitivity filter
3. sensitivity filter, thresholded (volume preserving)
4. density filter

Every design is saved as a separate png without axes/bounding box.
"""
from os.path import dirname, abspath, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.utils import map_eltoimg
from topoptlab.output_designs import threshold

NELX, NELY = 60, 20
VOLFRAC = 0.5
RMIN = 2.4
PENAL = 3.0


def run_mbb(ft: int) -> np.ndarray:
    """
    Run the 60x20 MBB beam with filter flag ft (-1 none, 0 sens., 1 dens.)
    and return the physical densities.
    """
    _, _, xPhys, _ = main(nelx=NELX, nely=NELY, volfrac=VOLFRAC,
                          rmin=RMIN, ft=ft,
                          matinterpol_kw={"eps": 1e-9, "penal": PENAL},
                          filter_mode="matrix",
                          optimizer="oc",
                          bcs=mbb_2d,
                          nouteriter=2000,
                          output_kw={"file": "mbb_filter",
                                     "display": False,
                                     "export": False,
                                     "write_log": False,
                                     "profile": False,
                                     "verbosity": 0,
                                     "output_movie": False,
                                     "save_pdf": False})
    return xPhys


def save_design(xPhys: np.ndarray, 
                file: str) -> None:
    """
    Save design as png without axes, ticks or bounding box.
    """
    img = map_eltoimg(quant=-xPhys, nelx=NELX, nely=NELY)
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.imshow(img, cmap="gray", interpolation="none",
              norm=Normalize(vmin=-1, vmax=0))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(file, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)
    return


if __name__ == "__main__":
    prefix = join(dirname(abspath(__file__)),
                  f"mbb_{NELX}x{NELY}_{RMIN:g}".replace(".", "p"))
    # 1. no filter
    save_design(run_mbb(ft=-1), f"{prefix}_nofilter.png")
    # 2. sensitivity filter and 3. its thresholded version
    xsens = run_mbb(ft=0)
    save_design(xsens, f"{prefix}_sensitivity.png")
    save_design(threshold(xsens, VOLFRAC), f"{prefix}_sensitivity_thresholded.png")
    # 4. density filter
    save_design(run_mbb(ft=1), f"{prefix}_density.png")
