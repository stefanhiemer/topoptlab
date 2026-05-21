# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.objectives import vol_frac, compliance
from topoptlab.filter.density_filter import DensityFilter

import sys

if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    compliance_limit = 300.
    rmin = 0.04*nelx
    penal = 3.0
    ft = DensityFilter
    display = True
    export = False
    write_log = True
    #
    if len(sys.argv)>1:
        nelx = int(sys.argv[1])
    if len(sys.argv)>2:
        nely = int(sys.argv[2])
    if len(sys.argv)>3:
        compliance_limit = float(sys.argv[3])
    if len(sys.argv)>4:
        rmin = float(sys.argv[4])
    if len(sys.argv)>5:
        penal = float(sys.argv[5])
    if len(sys.argv)>6:
        display = bool(int(sys.argv[6]))
    if len(sys.argv)>7:
        export = bool(int(sys.argv[7]))
    if len(sys.argv)>8:
        write_log = bool(int(sys.argv[8]))
    # initial guess
    
    #
    main(nelx=nelx, nely=nely,
         volfrac=None,
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         rmin=rmin,
         ft=ft,
         filter_mode="matrix",
         optimizer="mma",
         bcs=mbb_2d,
         obj_func=vol_frac,
         obj_kw={"scale_factor": 100},
         constraints=[{"name": "compliance",
                       "func": compliance,
                       "type": "leq",
                       "value": compliance_limit,
                       "kw":   {}}],
         output_kw={"file": "mbb_2d",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20,
                    "output_movie": False,
                    "save_pdf": True})
