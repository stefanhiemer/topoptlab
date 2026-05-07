# SPDX-License-Identifier: GPL-3.0-or-later
from topoptlab.topology_optimization import main
from topoptlab.optimizer.mma_utils import mma_defaultkws
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.accelerators import anderson
from topoptlab.filter.filter import TOFilter 
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.sensitivity_filter import SensitivityFilter
from topoptlab.filter.haeviside_projectors import HaevisideProjectorGuest2004,\
                                                  HaevisideProjectorSigmund2007,\
                                                  EtaProjectorXu2010
                                                    
                                                  
import numpy as np

if __name__ == "__main__":
    # Default input parameters
    nelx = 120
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 4.8  # 5.4
    penal = 3.0
    ft = [DensityFilter, 
          EtaProjectorXu2010] # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
    #
    accelerator_kw={"accel_freq": 4,
                    "accel_start": 50,
                    "max_history": 5,
                    "accelerator": anderson,
                    "damp": 0.9}
    write_log = True
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
    if len(sys.argv)>9:
        write_log = bool(int(sys.argv[9]))
    #
    optimizer_kw = mma_defaultkws(n=nelx*nely, 
                                  n_constr=1)
    optimizer_kw["move"] = 0.05
    optimizer_kw["asyincr"] = 1.05
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, 
                 matinterpol_kw={"eps":1e-9, "penal": penal},
                 rmin=rmin, 
                 ft=ft, 
                 l = 0.5,
                 filter_kw={"beta": 1,
                            "beta_scale": 2,
                            "beta_limit": 64,
                            "beta_update": 50, 
                            "volfrac": volfrac},
                 filter_mode="matrix",
                 optimizer="mma", optimizer_kw=optimizer_kw,
                 assembly_mode="full",
                 nouteriter=2000,
                 bcs=mbb_2d,
                 #body_forces_kw={"density_coupled": np.array([0,-0.01])},
                 #accelerator_kw=accelerator_kw,
                 output_kw = {"file": "mbb_2d",
                              "display": display,
                              "export": export,
                              "write_log": write_log,
                              "profile": False,
                              "verbosity": 20,
                              "output_movie": False})
