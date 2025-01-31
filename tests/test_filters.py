import numpy as np
from numpy.testing import assert_almost_equal
from scipy.ndimage import convolve

import pytest

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter

@pytest.mark.parametrize('nelx, nely, rmin, filter_mode, ndim',
                         [(10,10,2.4,"matrix",2),
                          (10,10,2.4,"convolution",2),])

def test_normalization(nelx,nely,rmin,filter_mode,ndim):
    np.random.seed(0)
    x = np.ones(nelx*nely)
    #
    desired = x.sum()
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,
                                      rmin=rmin,ndim=ndim)
        actual =np.asarray(H*x[np.newaxis].T/Hs)[:, 0].sum()
    elif filter_mode == "convolution":
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely, 
                                           rmin=rmin,ndim=ndim)
        actual = (convolve(x.reshape((nelx, nely)).T,
                           h,
                           mode="constant",
                           cval=0).T.flatten() / hs).sum()
    #
    assert_almost_equal(actual, desired)
    return

@pytest.mark.parametrize('nelx, nely, rmin, ndim',
                         [(10,10,2.4,2),
                          (20,10,2.4,2),
                          (10,20,2.4,2),])

def test_consistency(nelx,nely,rmin,ndim):
    np.random.seed(0)
    x = np.random.rand(nelx,nely).flatten()
    
    # matrix filter
    H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,
                                  rmin=rmin,ndim=ndim)
    desired =np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
    # convolution filter
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely, 
                                       rmin=rmin,ndim=ndim)
    actual = (convolve(x.reshape((nelx, nely)).T,
                       h,
                       mode="constant",
                       cval=0).T.flatten() / hs)
    #
    assert_almost_equal(actual, desired)
    return
    


    
