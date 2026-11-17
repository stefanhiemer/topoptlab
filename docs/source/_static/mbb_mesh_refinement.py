# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate MBB beam designs illustrating mesh independence of filtered designs:
60x20, 120x40 and 240x80 elements with rmin = 0.04*nelx (2.4, 4.8, 9.6),
each with sensitivity and density filter. For every design, a volume
preserving thresholded version is also saved.

Every design is saved as a separate png without axes/bounding box next to
this script.
"""
from os.path import dirname, abspath, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.utils import map_eltoimg
from topoptlab.output_designs import threshold

NELXS = [60, 120, 240]
VOLFRAC = 0.5
PENAL = 3.0
FILTERS = {0: "sensitivity", 1: "density"}


def run_mbb(nelx: int, nely: int, rmin: float, ft: int) -> np.ndarray:
    """
    Run the MBB beam with filter flag ft (0 sens., 1 dens.) and return the
    physical densities.
    """
    _, _, xPhys, _ = main(nelx=nelx, nely=nely, volfrac=VOLFRAC,
                          rmin=rmin, ft=ft,
                          matinterpol_kw={"eps": 1e-9, "penal": PENAL},
                          filter_mode="matrix",
                          optimizer="oc",
                          bcs=mbb_2d,
                          nouteriter=2000,
                          output_kw={"file": "mbb_mesh_refinement",
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
    Save design as png without axes, ticks or bounding box. Figure size is
    independent of the resolution so all designs have the same size.
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
        rmin = 0.04 * nelx
        for ft, name in FILTERS.items():
            prefix = join(outdir,
                          f"mbb_{nelx}x{nely}_{rmin:g}".replace(".", "p")
                          + f"_{name}")
            xPhys = run_mbb(nelx=nelx, nely=nely, rmin=rmin, ft=ft)
            save_design(xPhys, nelx, nely, f"{prefix}.png")
            save_design(threshold(xPhys, VOLFRAC), nelx, nely,
                        f"{prefix}_thresholded.png")
