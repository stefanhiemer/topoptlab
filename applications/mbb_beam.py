from topoptlab.compliance_minimization import main
from topoptlab.example_cases import mbb_2d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 5
    nely = 2#int(nelx/3)
    volfrac = 0.5
    rmin = 1.5  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode="matrix", solver="oc",nouteriter=1,
         bcs=mbb_2d,debug=True)
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode="convolution", solver="oc",nouteriter=1,
         bcs=mbb_2d,debug=True)
