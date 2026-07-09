# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.heat_conduction import heatplate_2d
from topoptlab.fem_solvers.heat_equation import HeatEquation

if __name__ == "__main__":
    nelx = 40
    nely = nelx
    volfrac = 0.4
    rmin = 0.03 * nelx
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens
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
    #
    solver = HeatEquation(nelx=nelx,
                          nely=nely,
                          bc=heatplate_2d,
                          ndim=2,
                          assembly_mode="full",
                          matinterpol_kw={"eps": 1e-9, "penal": penal})
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac,
         rmin=rmin,
         ft=ft, filter_mode="matrix", optimizer="oc", nouteriter=1000,
         matinterpol_kw={"eps": 1e-9, "penal": penal},
         problems=[solver],
         output_kw={"file": "heatplate_2d_solver",
                    "display": display,
                    "export": export,
                    "write_log": write_log,
                    "profile": False,
                    "verbosity": 20})
