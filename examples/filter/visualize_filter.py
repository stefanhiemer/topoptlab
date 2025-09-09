# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from topoptlab.filter.convolution_filter import assemble_convolution_filter
from topoptlab.filter.matrix_filter import assemble_matrix_filter
from topoptlab.filter.utils import visualise_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel


def initialize_matrix_filter(nelx,nely,nelz,rmin):
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                  rmin=rmin,ndim=ndim)
    return partial(apply_matrix_filter,
                   H=H,Hs=Hs)

def apply_matrix_filter(x,H,Hs):
    return np.asarray(H*x/Hs)

def initialize_convol_filter(nelx,nely,nelz,rmin):
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
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                       rmin=rmin,
                                       mapping=mapping,
                                       invmapping=invmapping)
    return partial(apply_convolution_filter,
                   h=h,hs=hs,
                   invmapping=invmapping,
                   mapping=mapping,
                   ndim=ndim)

def apply_convolution_filter(x,h,hs,invmapping,mapping,ndim):
    x_img = mapping(x)
    filtered_img = np.zeros(x_img.shape)
    convolve(x_img,
             weights=h, axes=(0,1,2)[:ndim],
             output=filtered_img,
             mode="constant",
             cval=0.0)
    return invmapping(filtered_img)/hs

if __name__ == "__main__":
    #
    rmin = 1.5
    nelx = 6
    nely = 6
    nelz = 6
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    ax = visualise_filter(n=(nelx,nely,nelz)[:ndim],
                          apply_filter=initialize_matrix_filter(nelx,nely,nelz,
                                                                rmin))
    #
    plt.show()
