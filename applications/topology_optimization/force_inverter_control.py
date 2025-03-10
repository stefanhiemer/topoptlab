from numpy import zeros

#from topoptlab.compliant_mechanisms import main
from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import forceinverter_2d
from topoptlab.objectives import var_squarederror

# The real main driver
if __name__ == "__main__":
    
    # Default input parameters
    nelx = 80
    nely = 80
    volfrac = 0.3
    rmin = 3.0#0.04*nelx  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    #
    l = zeros((2*(nelx+1)*(nely+1),1))
    l[2*nelx*(nely+1),0] = 1
    # 
    u0 = -0.3#l.copy() * 2
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         bcs=forceinverter_2d , obj_func=var_squarederror ,
         obj_kw={"l": l,"u0": u0},alpha=None,
         ft=ft, filter_mode="matrix",optimizer="mma",
         export=False)
    
