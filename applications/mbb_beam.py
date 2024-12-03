from topoptlab.compliance_minimization import main
from topoptlab.example_cases import mbb_2d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 0.04*nelx  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, passive=False,pde=False,solver="oc",
         bcs=mbb_2d)
