# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate MBB beam designs without any filter (optimized with MMA) for
60x20, 120x40 and 240x80 elements to illustrate checkerboard patterns and
that the design does not converge to a fixed length scale under mesh
refinement. Each design is saved as png without axes/bounding box next to
this script.
"""
from os.path import dirname, abspath, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.utils import map_eltoimg

NELXS = [60, 120, 240]
VOLFRAC = 0.5
PENAL = 3.0


def run_mbb(nelx: int, nely: int) -> np.ndarray:
    """
    Run the MBB beam without filter and return the physical densities.
    """
    _, _, xPhys, _ = main(nelx=nelx, nely=nely, volfrac=VOLFRAC,
                          rmin=0.04 * nelx, ft=-1,
                          matinterpol_kw={"eps": 1e-9, "penal": PENAL},
                          filter_mode="matrix",
                          optimizer="mma",
                          bcs=mbb_2d,
                          nouteriter=2000,
                          output_kw={"file": "mbb_checkerboard",
                                     "display": False,
                                     "export": False,
                                     "write_log": False,
                                     "profile": False,
                                     "verbosity": 0,
                                     "output_movie": False,
                                     "save_pdf": False})
    return xPhys


def save_design(xPhys: np.ndarray, nelx: int, nely: int, file: str) -> None:
    """
    Save design as png without axes, ticks or bounding box.
    """
    img = map_eltoimg(quant=-xPhys, nelx=nelx, nely=nely)
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.imshow(img, cmap="gray", interpolation="none",
              norm=Normalize(vmin=-1, vmax=0))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(file, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    outdir = dirname(abspath(__file__))
    for nelx in NELXS:
        nely = int(nelx / 3)
        save_design(run_mbb(nelx=nelx, nely=nely), nelx, nely,
                    join(outdir, f"mbb_{nelx}x{nely}_checkerboard.png"))
