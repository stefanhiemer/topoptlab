# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
import numpy as np

from topoptlab.topology_optimization import main
from topoptlab.example_bc.lin_elast import jaws_2d, gripper_2d
from topoptlab.objectives import var_maximization
from topoptlab.geometries import slab
from topoptlab.utils import check_meshdata, elid_to_coords, nodeid_to_coords
from topoptlab.optimizer.mma_utils import mma_defaultkws

if __name__ == "__main__":
    # Default input parameters
    nelx = 60
    nely = int(nelx/2)
    input_frac =  0.4
    output_frac = 0.3
    slit_height = 0.1
    #
    volfrac = 0.5
    rmin = 0.04*nelx  # 5.4
    #
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    display = True
    export = True
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
    optimizer_kw["move"] = 0.2
    #
    ndof = (nelx+1)*(nely+1)*2
    output_inds = int(slit_height*nely*2) + np.arange(0,ndof,2*(nely+1)) + 1
    output_inds = output_inds[-int(output_frac*nelx):]
    l = np.zeros((2*(nelx+1)*(nely+1),1))
    l[output_inds,0] = 1
    # create the slit
    pass_el = slab(nelx=nelx, nely=nely, 
                   center=(nelx*(1-output_frac/2),nely*slit_height/2 -1), 
                   fill_value=1, 
                   widths=[int(output_frac*nelx),
                   int(slit_height*nely)])
    # adapt boundary conditions
    bc = partial(jaws_2d, 
                 input_frac=input_frac, 
                 output_frac=output_frac, 
                 slit_height=slit_height)
    #
    main(nelx=nelx, nely=nely, volfrac=volfrac, 
         matinterpol_kw={"eps":1e-9, "penal": penal}, 
         el_flags = pass_el,
         rmin=rmin, 
         bcs=bc , 
         obj_func=var_maximization ,obj_kw={"l": l},
         ft=ft, filter_mode="matrix",
         optimizer="mma",
         optimizer_kw=optimizer_kw,
         nouteriter=2000,
         output_kw = {"file": "jaws_2d",
                      "display": display,
                      "export": export,
                      "write_log": write_log,
                      "profile": False,
                      "verbosity": 20,
                      "output_movie": False,
                      "save_pdf": True})
    
