# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_opt.driver import main
from topoptlab.fem_solvers.linear_elasticity import LinearElasticity
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.filter.filter import TOFilter
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.sensitivity_filter import SensitivityFilter

import numpy as np

if __name__ == "__main__":
    nelx = 60
    nely = int(nelx / 3)
    volfrac = 0.5
    rmin = 0.04 * nelx
    penal = 3.0
    ft = 0
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
    main(nelx=nelx, nely=nely, volfrac=volfrac,
         problems=[LinearElasticity],
         bcs=mbb_2d,
         solver_kw={"material_kw": {"Young's modulus": 1.0, "Poisson's ratio": 0.3}},
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         rmin=rmin,
         ft=ft,
         filter_kw={},
         filter_mode="matrix",
         optimizer="oc",
         assembly_mode="full",
         nouteriter=2000,
         output_kw={"file": "mbb_2d_elasticity",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20,
                    "output_movie": False,
                    "save_pdf": True})
