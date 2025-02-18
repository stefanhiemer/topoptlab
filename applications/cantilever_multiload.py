from topoptlab.compliance_minimization import main
from topoptlab.example_cases import cantilever_2d_twoloads_wrong,cantilever_2d_twoloads

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 150
    nely = 150
    volfrac = 0.4
    rmin = 6  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, 
         rmin=rmin, ft=ft, filter_mode = "matrix",
         optimizer="oc",solver="cvxopt-cholmod",
         bcs=cantilever_2d_twoloads_wrong,
         debug=False,display=False,export=False)
