# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.param_continuation import beta_scaling

if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    volfrac = 0.5
    rmin = 0.04*nelx  # 5.4
    penal = 3.0
    ft = 5#DensityFilter # ft==0 -> sens, ft==1 -> dens
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
    main(nelx=nelx, nely=nely, volfrac=volfrac, 
                 matinterpol_kw={"eps":1e-9, "penal": penal},
                 rmin=rmin, 
                 ft=ft, 
                 filter_kw={"baseplate": "S", 
                            "beta": 4,
                            "volfrac": volfrac},
                 filter_mode="convolution",
                 optimizer="oc",
                 assembly_mode="full",
                 nouteriter=2000,
                 bcs=mbb_2d,
                 output_kw = {"file": "mbb_2d",
                              "display": display,
                              "export": export,
                              "write_log": write_log,
                              "profile": False,
                              "verbosity": 20,
                              "output_movie": False,
                              "save_pdf": True}, 
                continuation_kw = {"funcs": [beta_scaling],
                                    "func_kws": [{"beta_limit": 64,
                                                  "state": {}}#,{"stage": 0}
                                                 ]})
