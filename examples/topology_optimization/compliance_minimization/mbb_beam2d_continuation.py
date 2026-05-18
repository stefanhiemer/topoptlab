# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from topoptlab.topology_optimization import main
from topoptlab.optimizer.mma_utils import mma_defaultkws
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.accelerators import anderson
from topoptlab.filter.filter import TOFilter 
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.sensitivity_filter import SensitivityFilter
from topoptlab.filter.haeviside_projectors import HaevisideProjectorGuest2004,\
                                                  HaevisideProjectorSigmund2007,\
                                                  EtaProjectorXu2010,\
                                                  MultiEtaProjectorXu2010
from topoptlab.convergence_criteria import max_design_change, norm_design_change
from topoptlab.param_continuation import adaptive_beta_continuation, update_move_limit

if __name__ == "__main__":
    # Default input parameters
    nelx = 120
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 0.02*nelx  # 5.4
    penal = 3.0
    ft = [DensityFilter,
          EtaProjectorXu2010,  # single-eta
          ] # ft==0 -> sens, ft==1 -> dens
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
    #optimizer_kw["asyincr"] = 1.05
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, 
                 matinterpol_kw={"eps":1e-9, "penal": penal},
                 rmin=rmin, 
                 ft=ft, 
                 l = 0.5,
                 filter_kw={"beta": 4,
                            "volfrac": volfrac,
                            "n_etas": 2,
                            "weights": np.array([0.7,0.3])
                            },
                 filter_mode="matrix",
                 optimizer="mma", optimizer_kw=optimizer_kw,
                 lin_solver_kw = {"name": "cvxopt-cholmod"},
                 assembly_mode="lower",
                 nouteriter=2000,
                 bcs=mbb_2d,
                 #body_forces_kw={"density_coupled": np.array([0,-0.01])},
                 #accelerator_kw=accelerator_kw,
                 output_kw = {"file": "mbb-continued_2d",
                              "display": display,
                              "export": export,
                              "write_log": write_log,
                              "profile": False,
                              "verbosity": 20,
                              "output_movie": False,
                              "save_pdf": True}, 
                 convergence_kw = {"conv_tol": 1e-2,
                                   "change_func": max_design_change,
                                   "ord": 2},
                 continuation_kw = {"funcs": [adaptive_beta_continuation#,update_move_limit,
                                              ],
                                    "func_kws": [{"beta_limit": 64,
                                                  "state": {}}#,{"stage": 0}
                                                 ]},
                 )
