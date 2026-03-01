from numpy import array,prod,ones
from numpy.random import seed,rand
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.stiffness_tensors import isotropic_2d
from topoptlab.elements.huhu_2d import _lk_huhu_2d, lk_huhu_2d
from topoptlab.elements.huhu_3d import _lk_huhu_3d, lk_huhu_3d
from topoptlab.elements.check_tangents import check_tangent_fd

@pytest.mark.parametrize('xe, kr, l, g',
                         [(array([[[-2,-1],[2,-1],[2,1],[-2,1]]]),
                           array([1.]),
                           array([4.,2.]), 
                           array([0.])),
                          (array([[[-2,-1,-1],
                                   [2,-1,-1],
                                   [2,1,-1],
                                   [-2,1,-1],
                                   [-2,-1,1],
                                   [2,-1,1],
                                   [2,1,1],
                                   [-2,1,1]]]), 
                           array([1.]), 
                           array([4.,2.,2.]), 
                           array([0.,0.]))
                          ,])

def test_linear(xe,kr,l,g):
    #
    ndim = xe.shape[-1]
    #
    if len(xe.shape) == 2:
        xe = xe[None,...]
    #
    if ndim == 2:
        _lk = _lk_huhu_2d
        lk = lk_huhu_2d
    elif ndim == 3:
        _lk = _lk_huhu_3d
        lk = lk_huhu_3d
    #
    assert_almost_equal(lk(kr=kr,l=l,g=g), 
                    _lk(xe=xe,ue=None,kr=kr,exponent=None))  
    return

@pytest.mark.parametrize('xe, u, a, kr',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2.1,-2],[2.1,-2],[2,2.1],[-2,2]]]),
                           array([[0.,0.,
                                   0.,0.,
                                   0.1,0.2,
                                   0.,0.],
                                  [0.,0.,
                                   0.,0.,
                                   0.1,0.2,
                                   0.,0.]]),
                           1.,
                           1.),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                   [[-2.1,-2,-2],[2.1,-2,-2],
                                    [2,2.1,-2],[-2,2,-2], 
                                    [-2.1,-2,2],[2.1,-2,2],
                                    [2,2.1,2],[-2,2,2]]]),
                            array([[0.,0.,0.,
                                    0.,0.,0.,
                                    0.1,0.2,0.15,
                                    0.,0.,0., 
                                    0.,0.,0.,
                                    0.,0.,0.,
                                    0.1,0.2,0.15,
                                    0.,0.,0.],
                                   [0.,0.,0.,
                                    0.,0.,0.,
                                    0.1,0.2,0.15,
                                    0.,0.,0.,
                                    0.,0.,0.,
                                    0.,0.,0.,
                                    0.1,0.2,0.15,
                                    0.,0.,0.]]),
                            1.,
                            1.),])

def test_tangents(xe,u,a,kr):
    #
    ndim = xe.shape[-1]
    #
    if len(xe.shape) == 2:
        xe = xe[None,...]
    #
    if ndim == 2:
        lk = _lk_huhu_2d
    elif ndim == 3:
        lk = _lk_huhu_3d
    #
    errs = check_tangent_fd(Ke_fe = lk,
                            Ke_fe_args = {"xe": xe,
                                          "mode": "newton",
                                          "exponent": a,
                                          "kr": kr},
                            u = u)
    assert errs.max() < 1e-7
    return