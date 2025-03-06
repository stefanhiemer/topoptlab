from functools import partial

import numpy as np
from numpy import ones,asarray
from numpy.random import seed,rand
from scipy.ndimage import convolve

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

def compare_filtering(x,nelx,nely,nelz,n,mapping,invmapping):
    
    #
    H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                  rmin=rmin,ndim=ndim)
    desired =asarray(H*x[None].T/Hs)[:, 0]
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz, 
                                       rmin=rmin,
                                       mapping=mapping,
                                       invmapping=invmapping)
    actual = invmapping(convolve(mapping(x),
                                 h,
                                 mode="constant",
                                 cval=0)) / hs
    print(mapping(x).shape)
    print(H.todense(),"\n",Hs)
    print()
    print(h,"\n",hs)
    print()
    #
    print(x,"\n",
          desired,"\n",
          actual,"\n",
          np.abs(desired-actual).max())
    return

if __name__ == "__main__":
    #
    rmin = 1.5
    nelx=30
    nely = 10
    nelz = 10
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
    x = np.ones(n)#np.arange(n)/(n-1)
    #compare_filtering(x,nelx,nely,nelz,n,mapping,invmapping)
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz, 
                                       rmin=rmin,
                                       mapping=mapping,
                                       invmapping=invmapping)
    # 
    ind = np.argmax(h,axis=0)[0]
    h[:,:] = 0
    h[ind[0],ind[1],ind[2]]=1.
    print(h)
    #
    actual = invmapping(convolve(mapping(x),
                                 h,
                                 mode="constant",
                                 cval=0))
    print(x,"\n",actual)
    