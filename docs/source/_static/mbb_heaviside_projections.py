# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generate the 60x20 MBB beam designs (rmin=2.4) with density filter followed
by one of three Heaviside projections:

1. Guest et al. 2004
2. Sigmund 2007
3. Xu et al. 2010 (eta projection, volume preserving)

The projection strength beta is increased by continuation. Every design is
saved as a separate png without axes/bounding box next to this script.
"""
from os.path import dirname, abspath, join

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.topology_optimization import main
from topoptlab.optimizer.mma_utils import mma_defaultkws
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.haeviside_projectors import HaevisideProjectorGuest2004,\
                                                  HaevisideProjectorSigmund2007,\
                                                  EtaProjectorXu2010
from topoptlab.convergence_criteria import norm_design_change
from topoptlab.param_continuation import beta_scaling
from topoptlab.utils import map_eltoimg

NELX, NELY = 60, 20
VOLFRAC = 0.5
RMIN = 2.4
PENAL = 3.0
PROJECTORS = {"guest2004": HaevisideProjectorGuest2004,
              "sigmund2007": HaevisideProjectorSigmund2007,
              "xu2010": EtaProjectorXu2010}


def run_mbb(projector) -> np.ndarray:
    """
    Run the MBB beam with density filter and the given Heaviside projector
    and return the physical densities.
    """
    optimizer_kw = mma_defaultkws(n=NELX*NELY, n_constr=1)
    optimizer_kw["move"] = 0.025
    _, _, xPhys, _ = main(nelx=NELX, nely=NELY, volfrac=VOLFRAC,
                          rmin=RMIN,
                          ft=[DensityFilter, projector],
                          filter_kw={"beta": 4, "volfrac": VOLFRAC},
                          matinterpol_kw={"eps": 1e-9, "penal": PENAL},
                          filter_mode="matrix",
                          optimizer="mma", optimizer_kw=optimizer_kw,
                          bcs=mbb_2d,
                          nouteriter=2000,
                          convergence_kw={"conv_tol": 1e-2,
                                          "change_func": norm_design_change,
                                          "ord": 2},
                          continuation_kw={"funcs": [beta_scaling],
                                           "func_kws": [{"beta_scale": 2,
                                                         "beta_limit": 64,
                                                         "beta_update": 100,
                                                         "state": {}}]},
                          output_kw={"file": "mbb_heaviside",
                                     "display": False,
                                     "export": False,
                                     "write_log": False,
                                     "profile": False,
                                     "verbosity": 0,
                                     "output_movie": False,
                                     "save_pdf": False})
    return xPhys


def save_design(xPhys: np.ndarray, file: str) -> None:
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


if __name__ == "__main__":
    prefix = join(dirname(abspath(__file__)),
                  f"mbb_{NELX}x{NELY}_{RMIN:g}".replace(".", "p"))
    for name, projector in PROJECTORS.items():
        save_design(run_mbb(projector), f"{prefix}_{name}.png")
