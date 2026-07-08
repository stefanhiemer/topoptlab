# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.filter.filter_bc import create_filter_bc

import numpy as np

if __name__ == "__main__":
    nelx = 60
    nely = int(nelx / 3)
    volfrac = 0.5
    rmin = 0.04 * nelx
    penal = 3.0
    ft = 1
    display = True
    export = False
    write_log = True

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

    ndof = 2 * (nelx + 1) * (nely + 1)
    _, f, fixed, _, _ = mbb_2d(nelx=nelx, nely=nely, ndof=ndof)

    el_flags = create_filter_bc(nelx=nelx,
                                nely=nely,
                                rmin=rmin,
                                fixed=fixed,
                                f=f,
                                mirror_sides=["l"],
                                vectorfield=True)

    main(nelx=nelx, nely=nely, volfrac=volfrac,
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         rmin=rmin,
         ft=ft,
         filter_kw={},
         filter_mode="matrix",
         optimizer="mma",
         assembly_mode="full",
         nouteriter=2000,
         bcs=mbb_2d,
         el_flags=el_flags,
         el_flags_policy={"correct_forward": True,
                          "correct_backward": True,
                          "neglect_in_filter": False},
         output_kw={"file": "mbb_2d_filter_bc",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20,
                    "output_movie": False,
                    "save_pdf": True})
