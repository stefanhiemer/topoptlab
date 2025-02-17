from topoptlab.compliance_minimization import main
from topoptlab.example_cases import heatplate_2d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40
    volfrac = 0.4
    rmin = 1.2
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode="matrix", optimizer="oc",nouteriter=1000,
         bcs=heatplate_2d,debug=False)
