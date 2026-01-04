# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.heat_conduction import heatplate_3d
from topoptlab.elements.poisson_3d import lk_poisson_3d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40#int(nelx/2)
    nelz = 40#int(nelx/2)
    volfrac = 0.4
    rmin = 0.03*nelx
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    display = False
    export = True
    write_log = True
    #
    import sys
    if len(sys.argv)>1:
        nelx = int(sys.argv[1])
    if len(sys.argv)>2:
        nely = int(sys.argv[2])
    if len(sys.argv)>3:
        nely = int(sys.argv[3])
    if len(sys.argv)>4:
        volfrac = float(sys.argv[4])
    if len(sys.argv)>5:
        rmin = float(sys.argv[5])
    if len(sys.argv)>6:
        penal = float(sys.argv[6])
    if len(sys.argv)>7:
        ft = int(sys.argv[7])
    if len(sys.argv)>8:
        display = bool(int(sys.argv[8]))
    if len(sys.argv)>9:
        export = bool(int(sys.argv[9]))
    if len(sys.argv)>10:
        write_log = bool(int(sys.argv[10]))
    #
    main(nelx=nelx, nely=nely, nelz=nelz, 
         volfrac=volfrac, 
         matinterpol_kw={"eps":1e-9, "penal": penal},
         rmin=rmin, 
         ft=ft, filter_mode="matrix", optimizer="oc",nouteriter=100,
         lin_solver_kw = {"name": "cvxopt-cholmod"}, assembly_mode="lower",
         #lin_solver_kw = {"name": "topoptlab-cg"}, preconditioner_kw = {"name": "pyamg-pyamg-ruge_stuben"},
         bcs=heatplate_3d, 
         lk=lk_poisson_3d,
         output_kw = {"file": "heatplate_3d",
                      "display": display,
                      "export": export,
                      "write_log": write_log,
                      "profile": False,
                      "verbosity": 20})
