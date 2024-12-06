from topoptlab.folding_mechanism import main

if __name__ == "__main__":
    # Default input parameters
    nelx = 600
    nely = int(nelx/3)
    volfrac = 0.4
    rmin = 12.0#0.04*nelx  # 5.4
    penal = 5.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, passive=False,pde=False,solver="oc",
         nouteriter=100)
