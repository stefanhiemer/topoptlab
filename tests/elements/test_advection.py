from numpy import array,stack,vstack,eye,tan

from numpy.testing import assert_almost_equal,assert_allclose

import pytest

from topoptlab.elements.advection import _lk_advection
from topoptlab.elements.advection_2d import lk_advection_2d
from topoptlab.elements.advection_3d import lk_advection_3d

@pytest.mark.parametrize('vs, xe, l, g',
                         [([array([1.,0.,1.,0.,1.,0.,1.,0.])],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2., 0.),
                          ([array([1.,0.,1.,0.,1.,0.,1.,0.]),
                            array([0.,1.,0.,1.,0.,1.,0.,1.])],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.2, 0.3),
                          ([array([1.,2.,3.,4.,5.,6.,7.,8.]),
                            array([-1.,1.,2.,-2.,0.5,-0.5,1.5,-1.5])],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2.4, -0.2),])

def test_advection_2d(vs,xe,l,g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l/2
    xe = xe@(R@S).T
    #
    Kes = stack([lk_advection_2d(v=v,l=array([l,l]),g=array([g])) for v in vs])
    #
    ve = stack(vs)
    assert_almost_equal(_lk_advection(xe=xe,ve=ve),
                        Kes)
    return

@pytest.mark.parametrize('vs, xe, l, g',
                         [([array([1.,0.,0.]*8)],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.2, [0.,0.]),
                          ([array([1.,0.,0.]*8),
                            array([0.,1.,0.]*8)],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           2., [0.1,-0.1]),
                          ([array([float(i) for i in range(24)]),
                            array([float(23-i) for i in range(24)])],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.5, [0.2,0.3]),])

def test_advection_3d(vs,xe,l,g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g[0])
    R[0,2] = tan(g[1])
    S = eye(3)*l/2
    xe = xe@(R@S).T
    #
    Kes = stack([lk_advection_3d(v=v,l=array([l,l,l]),g=array(g)) for v in vs])
    #
    ve = stack(vs)
    assert_almost_equal(_lk_advection(xe=xe,ve=ve),
                        Kes)
    return

@pytest.mark.parametrize('xe, ve',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]]),
                           array([[1.,0.,1.,0.,1.,0.,1.,0.],
                                  [0.,2.,0.,2.,0.,2.,0.,2.]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                   [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]),
                           array([[1.]*24,
                                  [2.]*24]))])

def test_consist(xe,ve):
    #
    Kes = vstack([_lk_advection(xe=xe[i:i+1],ve=ve[i:i+1]) for i in range(xe.shape[0])])
    #
    assert_allclose(_lk_advection(xe=xe,ve=ve),
                    Kes)
    return
