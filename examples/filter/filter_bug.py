# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial 

import numpy as np
from scipy.ndimage import convolve
from scipy.sparse import coo_array

from topoptlab.filters import assemble_matrix_filter,assemble_convolution_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel


if __name__ == "__main__":
    #
    rmin = 1.5
    nelx=5
    nely=2
    nelz=None
    #
    if nelz is None:
        ndim = 2
        n = nelx*nely
    else:
        ndim = 3
        n = nelx*nely*nelz
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
    dc = np.array([-705.65535777, -669.20970383, 
                   -412.529628, -447.82532496,
                   -232.57893823, -231.60167751, 
                   -107.68638974, -101.11611645,
                   -31.49861649, -126.63496201])
    #dc = np.arange(n,dtype=np.float64)
    
    H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,rmin=rmin)
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,rmin=rmin,
                                       mapping=mapping,invmapping=invmapping)
    
    
    test = np.asarray(H*(dc[None].T/Hs))[:, 0]
    test1 = invmapping(convolve(mapping(dc/hs),
                                h,
                                mode="constant",
                                cval=0.))
    print(Hs)
    print(hs)
    print("dc",dc)
    print("matrix",test)
    print("conv.",test1)
    print(test.sum(),test1.sum(),dc.sum())
    print(np.abs(test-test1).max())