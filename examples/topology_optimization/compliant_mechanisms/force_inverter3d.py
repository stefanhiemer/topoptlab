# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial

from numpy import zeros

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import forceinverter_3d
from topoptlab.objectives import var_maximization

if __name__ == "__main__":
    # Default input parameters
    nelx = 64
    nely = int(nelx/2)
    nelz = int(nelx/2)
    volfrac = 0.1
    rmin = 3*nelx/64 
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    display = False
    export = True
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
    l = zeros((3*(nelx+1)*(nely+1)*(nelz+1),1))
    l[3*nelx*(nely+1),0] = -1
    #
    main(nelx=nelx, nely=nely, nelz=nelz, 
         volfrac=volfrac, 
         penal=penal, rmin=rmin, 
         nouteriter=100,
         lin_solver="cvxopt-cholmod",
         assembly_mode="lower",
         bcs=partial(forceinverter_3d,fixation_mode="line") , 
         obj_func=var_maximization ,obj_kw={"l": l},
         ft=ft, filter_mode="matrix",optimizer="mma",
         output_kw = {"file": "force-inverter_3d",
                      "display": display,
                      "export": export,
                      "write_log": write_log,
                      "profile": False,
                      "debug": 0})
    
