# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from topoptlab.topology_opt.driver import main
from topoptlab.fem_solvers.linear_elasticity import LinearElasticity
from topoptlab.example_bc.lin_elast import forceinverter_2d
from topoptlab.objectives import var_maximization
from topoptlab.filter.density_filter import DensityFilter

if __name__ == "__main__":
    nelx = 40
    nely = int(nelx / 2)
    volfrac = 0.3
    rmin = 0.03 * nelx
    penal = 3.0
    ft = 1
    display = True
    export = False
    write_log = True
    #
    import sys
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
    #
    l = np.zeros((2 * (nelx + 1) * (nely + 1), 1))
    l[2 * nelx * (nely + 1), 0] = -1
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac,
         problems=[LinearElasticity],
         bcs=forceinverter_2d,
         solver_kw={"material_kw": {"Young's modulus": 1.0, "Poisson's ratio": 0.3}},
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         obj_func=var_maximization,
         obj_kw={"l": l},
         rmin=rmin,
         ft=ft,
         filter_kw={},
         filter_mode="convolution",
         optimizer="ocm",
         nouteriter=200,
         output_kw={"file": "force-inverter_2d_solver",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20,
                    "output_movie": False,
                    "save_pdf": True})
