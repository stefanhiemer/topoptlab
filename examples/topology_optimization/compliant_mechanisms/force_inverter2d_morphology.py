# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import forceinverter_2d
from topoptlab.objectives import var_maximization
from topoptlab.convergence_criteria import norm_design_change
from topoptlab.param_continuation import beta_scaling
from topoptlab.utils import map_eltoimg

if __name__ == "__main__":
    # Default input parameters
    nelx = 80
    nely = int(nelx/2)
    volfrac = 0.3
    rmin = 0.03*nelx
    penal = 3.0
    # morphology filters by Sigmund 2007 (see fetch_filters)
    fts = {"dilate": 6,
           "erode": 7,
           "open": 8,
           "close": 9}
    write_log = False
    #
    import sys
    if len(sys.argv)>1:
        nelx = int(sys.argv[1])
    if len(sys.argv)>2:
        nely = int(sys.argv[2])
    if len(sys.argv)>3:
        volfrac = float(sys.argv[3])
    if len(sys.argv)>4:
        rmin = float(sys.argv[4])
    if len(sys.argv)>5:
        penal = float(sys.argv[5])
    if len(sys.argv)>6:
        write_log = bool(int(sys.argv[6]))
    # indicator of the output dof (x displacement of the output node)
    l = np.zeros((2*(nelx+1)*(nely+1), 1))
    l[2*nelx*(nely+1), 0] = -1
    # run one optimization per filter and collect the physical designs
    designs = {}
    for name, ft in fts.items():
        x, xTilde, xPhys, obj = main(nelx=nelx, nely=nely, volfrac=volfrac,
                                     matinterpol_kw={"eps": 1e-9, "penal": penal},
                                     rmin=rmin,
                                     ft=ft,
                                     filter_kw={"beta": 1.},
                                     filter_mode="convolution",
                                     optimizer="mma",
                                     assembly_mode="full",
                                     nouteriter=1000,
                                     bcs=forceinverter_2d,
                                     obj_func=var_maximization,
                                     obj_kw={"l": l},
                                     output_kw = {"file": f"force-inverter_2d_{name}",
                                                  "display": False,
                                                  "export": False,
                                                  "write_log": write_log,
                                                  "profile": False,
                                                  "verbosity": 20,
                                                  "output_movie": False},
                                     convergence_kw = {"conv_tol": 1e-2,
                                                       "change_func": norm_design_change,
                                                       "ord": 2},
                                     continuation_kw = {"funcs": [beta_scaling],
                                                        "func_kws": [{"beta_scale": 2,
                                                                      "beta_limit": 64,
                                                                      "beta_update": 50,
                                                                      "state": {}}]},
                                     )
        designs[name] = (xPhys, obj)
    # display final designs. The objective is the negative output
    # displacement.
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (name, (xPhys, obj)) in zip(axs.flatten(), designs.items()):
        ax.imshow(map_eltoimg(xPhys, nelx=nelx, nely=nely)[:, :, 0],
                  cmap="gray_r", vmin=0., vmax=1.,
                  interpolation="none")
        ax.set_title(f"{name}: u_out = {-float(np.asarray(obj).ravel()[0]):.2f}, "
                     f"vol = {xPhys.mean():.2f}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
