# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.filter.filter_bc import create_filter_bc
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

if __name__ == "__main__":
    #
    nelx = 120
    nely = int(nelx / 3)
    nelz = None
    rmin = 0.04 * nelx

    ndof = 2 * (nelx + 1) * (nely + 1)
    _, f, fixed, _, _ = mbb_2d(nelx=nelx, nely=nely, ndof=ndof)

    el_flags = create_filter_bc(nelx=nelx,
                                nely=nely,
                                rmin=rmin,
                                fixed=fixed,
                                f=f,
                                mirror_sides=["l"],
                                vectorfield=True)
    #print()
    #print(el_flags)
    #
    if nelz is None:
        ndim = 2
    else:
        ndim=3
    #
    if ndim == 2:
        mapping = partial(map_eltoimg,
                          nelx=nelx,nely=nely)
        invmapping = partial(map_imgtoel,
                             nelx=nelx,nely=nely)
    elif ndim == 3:
        mapping = partial(map_eltovoxel,
                          nelx=nelx,nely=nely,nelz=nelz)
        invmapping = partial(map_voxeltoel,
                             nelx=nelx,nely=nely,nelz=nelz)
    #
    fig,ax = plt.subplots(1,1)
    ax.imshow(mapping(el_flags))
    plt.show()