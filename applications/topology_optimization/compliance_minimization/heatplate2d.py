from topoptlab.topology_optimization import main
from topoptlab.example_bc.heat_conduction import heatplate_2d
from topoptlab.elements.poisson_2d import lk_poisson_2d

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40
    volfrac = 0.4
    rmin = 1.2
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
    #
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        volfrac = float(sys.argv[3])
    if len(sys.argv)>4: 
        rmin = float(sys.argv[4])
    if len(sys.argv)>5: 
        penal = float(sys.argv[5])
    if len(sys.argv)>6: 
        ft = int(sys.argv[6])
    if len(sys.argv)>7:
        display = bool(int(sys.argv[7]))
    if len(sys.argv)>8:
        export = bool(int(sys.argv[8]))
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, filter_mode="matrix", optimizer="oc",nouteriter=1000,
         bcs=heatplate_2d, lk=lk_poisson_2d,
         file = "heatplate2d",
         debug=False, display=display, export=export)
