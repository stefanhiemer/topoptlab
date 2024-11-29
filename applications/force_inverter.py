from topoptlab.compliant_mechanisms import main

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 240
    nely = 40
    volfrac = 0.3
    rmin = 3.0#0.04*nelx  # 5.4
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
    main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
         ft=ft, passive=False,pde=True,solver="oc")
