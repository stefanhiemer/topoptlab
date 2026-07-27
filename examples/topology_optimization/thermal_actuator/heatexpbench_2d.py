# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import sys

from topoptlab.topology_opt.driver import main
from topoptlab.fem_solvers.linear_elasticity import LinearElasticity
from topoptlab.example_bc.lin_elast import heatexpbench_2d
from topoptlab.objectives import var_maximization

if __name__ == "__main__":
    # Default parameters
    nelx = 20
    nely = 50
    volfrac = 0.3
    rmin = 1.5
    penal = 3.0
    ft = 1  # density filter
    display = True
    export = False
    write_log = True

    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        rmin = float(sys.argv[4])
    if len(sys.argv) > 5:
        penal = float(sys.argv[5])
    if len(sys.argv) > 6:
        ft = int(sys.argv[6])
    if len(sys.argv) > 7:
        display = bool(int(sys.argv[7]))
    if len(sys.argv) > 8:
        export = bool(int(sys.argv[8]))
    if len(sys.argv) > 9:
        write_log = bool(int(sys.argv[9]))

    # Output indicator: maximize y-displacement at spring DOF (DOF 1, top-left node)
    ndof = 2 * (nelx + 1) * (nely + 1)
    l = np.zeros((ndof, 1))
    l[1, 0] = 1

    main(nelx=nelx, nely=nely, volfrac=volfrac,
         problems=[LinearElasticity],
         bcs=heatexpbench_2d,
         solver_kw={"material_kw": {"Young's modulus": 1.0,
                                    "Poisson's ratio": 0.3},
                    "thermal_expansion": True,
                    "thermal_expansion_kw": {"thermal expansion tensor": np.eye(2)*1e-3,
                                             "delta_T": 1.0}},
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         obj_func=var_maximization,
         obj_kw={"l": l},
         rmin=rmin,
         ft=ft,
         filter_kw={},
         filter_mode="convolution",
         optimizer="mma",
         nouteriter=2000,
         output_kw={"file": "heatexpbench_2d",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20,
                    "output_movie": False,
                    "save_pdf": True})
