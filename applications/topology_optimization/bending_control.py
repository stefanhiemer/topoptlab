from numpy import zeros,arange

#from topoptlab.compliant_mechanisms import main
from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import selffolding_2d
from topoptlab.objectives import var_squarederror

# The real main driver
if __name__ == "__main__":
    
    # Default input parameters
    nelx = 60
    nely = 20
    volfrac = 0.3
    rmin = 2.4#0.04*nelx  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    #
    l = zeros((2*(nelx+1)*(nely+1),1))
    l[arange(0,2*(nelx+1)*(nely+1),2*(nely+1))+1,0] = 1
    # 
    u0 = -0.3 * arange(0,nelx+1) * 1e-1#l.copy() * 2
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         bcs=selffolding_2d , obj_func=var_squarederror ,
         obj_kw={"l": l,"u0": u0},alpha=None,
         ft=ft, filter_mode="matrix",optimizer="mma",
         export=False)
    
