# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import cantilever_2d

# 
if __name__ == "__main__":
    # Default input parameters
    nelx = 160
    nely = int(5/8 * nelx)
    volfrac = 0.4
    rmin = 3/80 *nelx  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = True
    write_log = True
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
        ft = int(sys.argv[6])
    if len(sys.argv)>7:
        display = bool(int(sys.argv[7]))
    if len(sys.argv)>8:
        export = bool(int(sys.argv[8]))
    if len(sys.argv)>9:
        write_log = bool(int(sys.argv[9]))
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode = "matrix",optimizer="oc",
         bcs=cantilever_2d,
         output_kw = {"file": "cantilever_2d",
                      "display": display,
                      "export": export,
                      "write_log": write_log,
                      "profile": False,
                      "verbosity": 20, 
                      "output_movie": False})
