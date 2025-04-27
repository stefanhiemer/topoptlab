from topoptlab.compliance_minimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.accelerators import anderson

import numpy as np
# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 150
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    accelerator_kw={"accel_freq": 4, 
                    "accel_start": 100,
                    "max_history": 5,
                    "accelerator": anderson,
                    "damp": 0.9}
    x,obj = main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, 
                 rmin=rmin, ft=ft, filter_mode="matrix", 
                 optimizer="oc", lin_solver="scipy-direct",
                 nouteriter=1000,file="mbb_2d",
                 bcs=mbb_2d,debug=False,display=True,export=False)
   #main(nelx=nelx, nely=nely, volfrac=volfrac, penal=penal, rmin=rmin, 
   #      ft=ft, filter_mode="convolution", optimizer="oc",nouteriter=1000,
   #      bcs=mbb_2d,debug=False,export=False)
