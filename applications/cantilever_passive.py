from topoptlab.compliance_minimization import main
from topoptlab.example_cases import cantilever_2d
from topoptlab.geometries import sphere

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = 20
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    center = (nelx/3 - 1, nely/2 - 1)
    radius = nely/3
    pass_el = sphere(nelx, nely, center, radius)
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, el_flags = pass_el ,filter_mode = "matrix",optimizer="oc",
         bcs=cantilever_2d,debug=False,display=True)
