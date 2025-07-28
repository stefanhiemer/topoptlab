from functools import partial
from numpy import ones,asarray
from numpy.random import seed,rand
from numpy.testing import assert_almost_equal
from scipy.ndimage import convolve
from scipy.sparse import spmatrix,sparray

import pytest

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel


@pytest.mark.parametrize('nelx, nely, nelz, rmin, filter_mode',
                         [(10,10,None,2.4,"matrix"),
                          (10,10,None,2.4,"convolution"),
                          (10,10,10,2.4,"matrix"),
                          (10,10,10,2.4,"convolution"),])

def test_normalization(nelx,nely,nelz,rmin,filter_mode):
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
    x = ones((n,1),order="F")
    #
    desired = x.sum()
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                      rmin=rmin,ndim=ndim)
        if isinstance(H,spmatrix):
            actual = asarray(H*x/Hs)
        elif isinstance(H,sparray):
            actual = H @ x / Hs
    elif filter_mode == "convolution":
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                           rmin=rmin,
                                           mapping=mapping,
                                           invmapping=invmapping)
        actual = invmapping(convolve(mapping(x),
                                     h,
                                     mode="constant",axes=(0,1,2)[:ndim],
                                     cval=0)) / hs
    actual = actual.sum()
    #
    assert_almost_equal(actual, desired)
    return

@pytest.mark.parametrize('nelx, nely, nelz, rmin',
                         [(10,10,None,2.4),
                          (20,10,None,2.4),
                          (10,20,None,2.4),
                          (10,10,10,2.4),
                          (20,10,10,2.4),
                          (10,20,10,2.4),])

def test_consistency(nelx,nely,nelz,rmin):
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
    seed(0)
    x = rand(n,1).flatten(order="F")
    # matrix filter
    H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                  rmin=rmin,ndim=ndim)

    if isinstance(H,spmatrix):
        desired = asarray(H*x/Hs)
    elif isinstance(H,sparray):
        desired = H @ x / Hs
    # convolution filter
    h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                       rmin=rmin,
                                       mapping=mapping,
                                       invmapping=invmapping)
    actual = invmapping(convolve(mapping(x),
                                 h,
                                 mode="constant",axes=(0,1,2)[:ndim],
                                 cval=0)) / hs
    #
    assert_almost_equal(actual, desired)
    return

from topoptlab.filters import find_eta, eta_projection

@pytest.mark.parametrize('n, beta, volfrac',
                         [(10,10,0.3),
                          (10,1,0.5),])

def test_volume_conservation(n,beta,volfrac):
    #
    seed(0)
    x = rand(n)
    #
    assert_almost_equal(eta_projection(xTilde=x,
                                       eta=find_eta(xTilde=x,
                                                    beta=beta,
                                                    eta0=0.5,
                                                    volfrac=volfrac),
                                       beta=beta).mean(),
                        volfrac)
    return
