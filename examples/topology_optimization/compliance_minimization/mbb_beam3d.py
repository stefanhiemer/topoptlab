# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_3d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    nelz = int(nelx/6)
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    display = False
    export = False
    write_log=True
    #
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz = int(sys.argv[3])
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
    main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, 
         rmin=rmin, ft=ft, filter_mode="matrix", 
         optimizer="oc", lin_solver_kw = {"name": "cvxopt-cholmod"},
         assembly_mode="lower",
         nouteriter=5,
         bcs=mbb_3d,
         output_kw = {"file": "mbb_3d",
                      "display": display,
                      "export": export,
                      "write_log": write_log,
                      "profile": True,
                      "verbosity": 20})
    #main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, rmin=rmin, 
    #     ft=ft, filter_mode="convolution", optimizer="oc",nouteriter=1000,
    #     bcs=mbb_3d,
    #     debug=False,display=False,
    #     export=False)
