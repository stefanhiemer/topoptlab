from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter 
from topoptlab.filters import visualise_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel 


def prepare_filter(nelx,nely,nelz,rmin):
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
                   mapping=mapping)

def apply_convolution_filter(x,h,hs,invmapping,mapping):
    
    return invmapping(convolve(mapping(x),
                          weights=h,
                          mode="constant",
                          cval=0)) / hs

if __name__ == "__main__":
    #
    rmin = 1.5
    nelx=30
    nely = 10
    nelz = None 
    ax = visualise_filter(n=(nelx,nely), 
                          apply_filter=prepare_filter(nelx,nely,nelz,rmin))
    #
    plt.show()
    