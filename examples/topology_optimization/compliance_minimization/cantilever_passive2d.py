# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import cantilever_2d
from topoptlab.geometries import sphere

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = 20
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
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
    center = (nelx/3 - 1, nely/2 - 1)
    radius = nely/3
    pass_el = sphere(nelx=nelx, nely=nely, center=center, radius=radius)
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin,
         ft=ft, el_flags = pass_el ,filter_mode = "matrix",optimizer="oc",
         bcs=cantilever_2d,
         debug=False,display=display,export=export)
