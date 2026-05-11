from functools import partial
from numpy import ones, asarray
from numpy.random import seed,rand
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.ndimage import convolve
from scipy.sparse import spmatrix,sparray

import pytest

from topoptlab.filter.convolution_filter import assemble_convolution_filter 
from topoptlab.filter.matrix_filter import assemble_matrix_filter
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.sensitivity_filter import SensitivityFilter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.topology_optimization import main

@pytest.mark.parametrize('nelx, nely, volfrac, ft_int, ft_obj, rmin, filter_mode, bcs',
                         [(10,5,0.5,0,SensitivityFilter,1.5,"matrix",mbb_2d),
                          (10,5,0.5,1,DensityFilter,1.5,"matrix",mbb_2d),
                          (10,5,0.5,0,SensitivityFilter,1.5,"helmholtz",mbb_2d),
                          (10,5,0.5,1,DensityFilter,1.5,"helmholtz",mbb_2d),])

def test_filterobj(nelx, nely, volfrac, ft_int, ft_obj, rmin, filter_mode, bcs):
    """
    Test the minimum compliance problem with different filter settings. 
    Does exactly the same as function below. Just to allow to have fast and 
    slow tests in same file.
    """
    #
    x_int, _, _, obj_int = main(nelx=nelx, nely=nely, volfrac=volfrac, 
                  rmin=rmin, ft=ft_int, filter_mode=filter_mode,
                  optimizer="oc",
                  bcs=bcs,
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    x_obj, _, _, obj_obj = main(nelx=nelx, nely=nely, volfrac=volfrac, 
                  rmin=rmin, ft=ft_obj, filter_mode=filter_mode,
                  optimizer="oc",
                  bcs=bcs,
                  output_kw = {"file": None,
                               "display": False,
                               "export": False,
                               "write_log": False,
                               "profile": False,
                               "debug": 0})
    #
    assert_almost_equal(obj_int,obj_obj,decimal=8)
    assert_allclose(x_int,x_obj)
    return 

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

from topoptlab.filter.haeviside_projection import find_eta, eta_projection,\
                                                  find_multieta, multieta_projection

@pytest.mark.parametrize('n, beta, volfrac, filter, finder, kwargs',
                         [(100,10,0.3,eta_projection,find_eta,
                           {'eta0': 0.5}),
                          (100,1,0.5,eta_projection,find_eta,
                           {'eta0': 0.5}),
                          (100,10,0.3,multieta_projection,find_multieta,
                           {'etas0': [0.3,0.7], 
                            "weights": asarray([0.3,0.7])}),
                          (100,1,0.5,multieta_projection,find_multieta,
                           {'etas0': [0.3,0.7], 
                            "weights": asarray([0.3,0.7])})])

def test_volume_conservation(n,beta,volfrac,filter,finder,kwargs):
    #
    seed(0)
    x = rand(n)
    #
    params = finder(xTilde=x,
                    beta=beta,
                    volfrac=volfrac,
                    **kwargs)
    #
    if filter is multieta_projection:
        xPhys = filter(xTilde=x,
                       etas=params,
                       beta=beta, 
                       **kwargs)
    else:
        xPhys = filter(xTilde=x,
                       eta=params,
                       beta=beta)
    #
    assert_almost_equal(xPhys.mean(),
                        volfrac)
    return
