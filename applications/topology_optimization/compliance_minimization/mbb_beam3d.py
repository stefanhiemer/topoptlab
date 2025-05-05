from topoptlab.compliance_minimization import main
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
    display = True
    export = False
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
    #
    main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, 
         rmin=rmin, ft=ft, filter_mode="matrix", 
         optimizer="oc", lin_solver = "cvxopt-cholmod",
         nouteriter=2000,
         bcs=mbb_3d,
         file="mbb_3d",
         debug=False,display=display,
         export=export)
    #main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, rmin=rmin, 
    #     ft=ft, filter_mode="convolution", optimizer="oc",nouteriter=1000,
    #     bcs=mbb_3d,
    #     debug=False,display=False,
    #     export=False)
