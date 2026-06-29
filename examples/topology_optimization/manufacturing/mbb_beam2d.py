# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.param_continuation import beta_scaling
from topoptlab.objectives import am_angle
from topoptlab.optimizer.mma_utils import mma_defaultkws


if __name__ == "__main__":
    # Default input parameters
    nelx = 120
    nely = int(nelx/3)
    nelz=None
    volfrac = 0.5
    rmin = 3.6  # 5.4
    penal = 3.0
    ft = 1#DensityFilter # ft==0 -> sens, ft==1 -> dens
    display = True
    export = False
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
    x, xTilde, xPhys, obj = main(nelx=nelx, nely=nely, volfrac=volfrac, 
                                 matinterpol_kw={"eps":1e-9, 
                                                 "penal": penal},
                                 rmin=rmin, 
                                 ft=1, 
                                 filter_kw={},
                                 filter_mode="convolution",
                                 optimizer="mma",
                                 assembly_mode="full",
                                 nouteriter=50,
                                 bcs=mbb_2d,
                                 output_kw = {"file": "pre-mbb_2d-manufacturing-constraint",
                                              "display": display,
                                              "export": export,
                                              "write_log": write_log,
                                              "profile": False,
                                              "verbosity": 20,
                                              "output_movie": False,
                                              "save_pdf": True})
    #
    if nelz is None:
        ndim = 2 
    else:
        ndim = 3
    #
    optimizer_kw = mma_defaultkws(n=int(np.prod([nelx,nely,nelz][:ndim])), 
                                  n_constr=2)
    optimizer_kw["move"] = 0.025
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, 
        matinterpol_kw={"eps":1e-9, "penal": penal},
        rmin=rmin, 
        ft=4, 
        obj_kw={"scale_factor": 1e-4},
        filter_kw={"beta": 8, 
                   "volfrac": None, 
                   "eta": 0.5},
        filter_mode="convolution",
        optimizer="mma",
        optimizer_kw=optimizer_kw,
        lin_solver_kw = {"name": "cvxopt-cholmod"}, 
        assembly_mode="lower",
        nouteriter=2000,
        initial_guess={"xPhys": xPhys,  
                       "x": x,  
                       "xTilde-0": xTilde},
        bcs=mbb_2d,
        constraints=[{"name": "am_angle",
                      "func": am_angle,
                      "type": "leq",
                      "norm_ref": 1e-4,
                      "value": 0.,
                      "kw":   {"baseplate": "S", 
                               "ndim": 2,
                               "p": 10., 
                               "ks_exponent": 20.}}],
        output_kw = {"file": "mbb_2d-manufacturing-constraint",
                     "display": display,
                     "export": export,
                     "write_log": write_log,
                     "profile": False,
                     "verbosity": 20,
                     "output_movie": False,
                     "save_pdf": True}, 
        continuation_kw = {"funcs": [beta_scaling,],
                           "func_kws": [{"beta_limit": 16,
                                         "beta_scale": 1.1, 
                                         "beta_update": 100,
                                         "state": {}}]})
