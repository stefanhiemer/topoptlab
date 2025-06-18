from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.accelerators import anderson
import numpy as np

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
    #
    accelerator_kw={"accel_freq": 4,
                    "accel_start": 100,
                    "max_history": 5,
                    "accelerator": anderson,
                    "damp": 0.9}
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
    x,obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal,
                 rmin=rmin, ft=ft, filter_mode="matrix",
                 optimizer="mma", lin_solver="scipy-direct",
                 nouteriter=1000,file="mbb_2d",
                 bcs=mbb_2d, #body_forces_kw={"density_coupled": np.array([0,-0.01])},
                 debug=False,display=display,export=export)
