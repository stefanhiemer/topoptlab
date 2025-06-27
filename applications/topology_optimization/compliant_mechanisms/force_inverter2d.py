from numpy import zeros

from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import forceinverter_2d
from topoptlab.objectives import var_maximization

if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 20
    volfrac = 0.3
    rmin = 1.2#0.04*nelx  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
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
    l = zeros((2*(nelx+1)*(nely+1),1))
    l[2*nelx*(nely+1),0] = -1
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         bcs=forceinverter_2d , obj_func=var_maximization ,obj_kw={"l": l},
         ft=ft, filter_mode="matrix",optimizer="ocm",
         display=display,export=export)
    
