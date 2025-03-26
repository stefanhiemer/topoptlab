from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import cantilever_2d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 160
    nely = 100
    volfrac = 0.4
    rmin = 6  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode = "matrix",optimizer="oc",
         bcs=cantilever_2d,debug=False,display=True)
