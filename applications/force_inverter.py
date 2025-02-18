from numpy import zeros

#from topoptlab.compliant_mechanisms import main
from topoptlab.compliance_minimization import main
from topoptlab.example_cases import forceinverter_2d
from topoptlab.objectives import var_maximization

# The real main driver
if __name__ == "__main__":
    
    # Default input parameters
    nelx = 2
    nely = 2
    volfrac = 0.3
    rmin = 3.0#0.04*nelx  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    #
    l = zeros((2*(nelx+1)*(nely+1),1))
    l[2*nelx*(nely+1),0] = -1
    #
    from topoptlab.compliant_mechanisms import main as main2
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         bcs=forceinverter_2d , obj_func=var_maximization ,obj_kw={"l": l},
         ft=ft, filter_mode="matrix",optimizer="ocm",
         export=False)
    
    x, obj = main2(nelx=nelx, nely=nely, 
                   volfrac=0.3, penal=3.0, rmin=rmin, ft=ft, 
                   passive=False,filter_mode="matrix", solver="oc",
                   display=True,export=False,write_log=True)
    
