from numpy import array,stack,eye,pi,tan
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d,lk_linear_elast_2d
from topoptlab.stiffness_tensors import isotropic_2d

@pytest.mark.parametrize('Es, nus, c, xe, l, g',
                         [([1.],[0.3],isotropic_2d(),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2,0.),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.5,0.),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           3,0.),
                          ([1.,2.],[0.3,0.4],
                           stack([isotropic_2d(1.0,0.3),isotropic_2d(2.,0.4)]),
                           array([[-1,-1],[1,-1],[1,1],[-1,1]]),
                           2,0.),])

def test_isotrop_linelast_2d(Es,nus,c,xe,l,g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l
    xe = xe@(R@S).T
    #
    Kes = stack([lk_linear_elast_2d(E=E,nu=nu,l=array([l,l]),g=array([g]))\
                 for E,nu in zip(Es,nus)])
    #
    assert_almost_equal(_lk_linear_elast_2d(xe=xe,c=c),
                        Kes)
    return

from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_aniso_2d

@pytest.mark.parametrize('c, xe, l, g',
                         [(array([[1,2,0],[2,1,0],[0,0,1]]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2,0.),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.5,0.),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.3,0.),
                          (stack([array([[1,2,0.5],[2,1,4],[0.5,4,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           array([[-1,-1],[1,-1],[1,1],[-1,1]]),
                           1.2,0.),])

def test_anisotrop_linelast_2d(c,xe,l,g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l
    xe = xe@(R@S).T
    #
    if len(c.shape) == 2:
        Kes = lk_linear_elast_aniso_2d(c=c,l=array([l,l]),g=array([g]))[None,:,:]
    else:
        Kes = stack([lk_linear_elast_aniso_2d(c=c[i],l=array([l,l]),g=array([g])) \
                     for i in range(c.shape[0])])
    #
    assert_almost_equal(_lk_linear_elast_2d(xe=xe,c=c),
                        Kes)
    return

from topoptlab.stiffness_tensors import isotropic_3d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d,lk_linear_elast_3d

@pytest.mark.parametrize('Es, nus, cs, xe, l, g',
                         [([1.],[0.3],isotropic_3d(E=1.,nu=0.3),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           2.,1.),
                          ([1.2],[0.3],isotropic_3d(E=1.2,nu=0.3),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.1,pi/4),
                          ([1.2,2.],[0.3,0.35],
                           stack([isotropic_3d(E=1.2,nu=0.3),
                                  isotropic_3d(E=2.,nu=0.35)]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.2,0.3)])

def isotrop_linelast_3d(E,nu,cs,xe,l,g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l
    xe = xe@(R@S).T
    #
    if len(cs.shape) == 2:
        Kes = lk_linear_elast_3d(E=E[0],nu=nu[0],
                                 l=array([l,l,l]),g=array([g,g]))[None,:,:]
    else:
        Kes = stack([lk_linear_elast_3d(E=_E,nu=_nu,
                                        l=array([l,l,l]),g=array([g,g])) \
                     for _E,_nu in zip(E,nu)])
    #
    assert_almost_equal(_lk_linear_elast_3d(xe=xe,c=cs),
                        Kes)
    return

from topoptlab.stiffness_tensors import isotropic_3d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_aniso_3d

@pytest.mark.parametrize('c, xe, l, g',
                         [(eye(6),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           2.,1.),
                          (isotropic_3d(),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.1,pi/3),
                          (stack([isotropic_3d(),2*isotropic_3d()]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.2,pi/4)])

def anisotrop_linelast_3d(cs,xe,l,g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l
    xe = xe@(R@S).T
    #
    if len(cs.shape) == 2:
        Kes = lk_linear_elast_aniso_3d(c=cs,l=array([l,l,l]),g=array([g,g]))[None,:,:]
    else:
        Kes = stack([lk_linear_elast_aniso_3d(c=c,l=array([l,l,l]),g=array([g,g])) \
                     for c in cs])
    #
    assert_almost_equal(_lk_linear_elast_3d(xe=xe,c=cs),
                        Kes)
    return
