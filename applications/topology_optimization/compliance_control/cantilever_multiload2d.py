from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import cantilever_2d_twoloads_wrong,cantilever_2d_twoloads
from topoptlab.objectives import compliance_control

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 150
    nely = 150
    volfrac = 0.4
    rmin = 6  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
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
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
         rmin=rmin,
         obj_func=compliance_control, obj_kw={"c0": 70},
         ft=ft, filter_mode = "matrix",
         optimizer="mma",lin_solver="cvxopt-cholmod",
         bcs=cantilever_2d_twoloads_wrong,
         debug=False,display=display,export=export)
