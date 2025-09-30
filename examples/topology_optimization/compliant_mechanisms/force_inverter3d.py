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
    rmin = 3*nelx/64 #0.04*nelx  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    display = False
    export = True
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
         file="force-inverter_3d",
         display=display,export=export)
    
