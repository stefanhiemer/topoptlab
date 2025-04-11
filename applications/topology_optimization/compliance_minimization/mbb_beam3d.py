from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d,mbb_3d

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
    main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, 
         rmin=rmin, ft=ft, filter_mode="matrix", 
         optimizer="oc", lin_solver = "cvxopt-cholmod",
         nouteriter=2000,
         bcs=mbb_3d,
         file="mbb_3d",
         debug=False,display=False,
         export=True)
    #main(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, penal=penal, rmin=rmin, 
    #     ft=ft, filter_mode="convolution", optimizer="oc",nouteriter=1000,
    #     bcs=mbb_3d,
    #     debug=False,display=False,
    #     export=False)
