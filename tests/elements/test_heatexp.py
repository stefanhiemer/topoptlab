from numpy import array,stack,eye,tan
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.heatexpansion_2d import fk_heatexp_2d,_fk_heatexp_2d
from topoptlab.stiffness_tensors import isotropic_2d

@pytest.mark.parametrize('Es, nus, _as, xe, l, g',
                         [([1.],[0.3],[0.05],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2.4,0.),
                          ([1.,2.],[0.3,0.4],[0.05,0.1],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.4,0.),
                          ([1.,2.],[0.3,0.4],[0.05,0.1],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           0.4,0.),])

def test_isotrop_heatexp_2d(Es,nus,_as,xe, l, g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l
    xe = xe@(R@S).T
    #
    Kes = stack([fk_heatexp_2d(E=E,nu=nu,a=a,l=array([l,l]),g=array([g])) \
                 for E,nu,a in zip(Es,nus,_as)])
    #
    cs = stack([isotropic_2d(E=E,nu=nu) for E,nu in zip(Es,nus)])
    _as = stack([a*eye(2) for a in _as])
    #
    assert_almost_equal(_fk_heatexp_2d(xe=xe,c=cs,a=_as),
                        Kes)
    return


from topoptlab.elements.heatexpansion_2d import fk_heatexp_aniso_2d

@pytest.mark.parametrize('cs, _as, xe, l, g',
                         [(array([[1,2,0],[2,1,0],[0,0,1]]),
                           eye(2)*0.05,
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2.4,0.),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           stack([eye(2)*0.05,eye(2)*0.1]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.5,0.),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           stack([eye(2)*0.05,eye(2)*0.1]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           .4,0.),
                          (stack([array([[1,2,0.5],[2,1,4],[0.5,4,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           stack([eye(2)*0.05,eye(2)*0.1]),
                           array([[-1,-1],[1,-1],[1,1],[-1,1]]),
                           2.4,0.),])

def test_anisotrop_heatexp_2d(cs,_as, xe, l, g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l
    xe = xe@(R@S).T
    #
    if len(cs.shape) == 2:
        Kes = fk_heatexp_aniso_2d(c=cs,a=_as,l=array([l,l]),g=array([g]))[None,:,:]
    else:
        Kes = stack([fk_heatexp_aniso_2d(c=cs[i],a=_as[i],l=array([l,l]),g=array([g])) for i in range(cs.shape[0])])
    #
    assert_almost_equal(_fk_heatexp_2d(xe=xe,c=cs,a=_as),
                        Kes)
    return

from topoptlab.elements.heatexpansion_3d import fk_heatexp_3d,_fk_heatexp_3d
from topoptlab.stiffness_tensors import isotropic_3d

@pytest.mark.parametrize('Es, nus, _as, xe, l, g',
                         [([1.],[0.3],[0.05],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           2.4,0.),
                          ([1.,2.],[0.3,0.4],[0.05,0.1],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.3,0.),
                          ([1.,2.],[0.3,0.4],[0.05,0.1],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           0.9,0.),])

def test_isotrop_heatexp_3d(Es,nus,_as,xe, l, g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l
    xe = xe@(R@S).T
    #
    Kes = stack([fk_heatexp_3d(E=E,nu=nu,a=a,l=array([l,l,l]),g=array([g,g])) for E,nu,a in zip(Es,nus,_as)])
    #
    cs = stack([isotropic_3d(E=E,nu=nu) for E,nu in zip(Es,nus)])
    _as = stack([a*eye(3) for a in _as])
    #
    assert_almost_equal(_fk_heatexp_3d(xe=xe,c=cs,a=_as),
                        Kes)
    return

from topoptlab.elements.heatexpansion_3d import fk_heatexp_aniso_3d


@pytest.mark.parametrize('cs, _as, xe, l, g',
                         [(array([[10,  5,  9,  4,  6,  4],
                                  [ 5,  2,  5,  2,  3,  6],
                                  [ 9,  5, 10,  2,  4,  1],
                                  [ 4,  2,  2,  2,  4,  6],
                                  [ 6,  3,  4,  4,  8,  7],
                                  [ 4,  6,  1,  6,  7,  2]]),
                           array([[ 3,  8,  9],
                                  [ 8, 10,  7],
                                  [ 9,  7,  3]]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.5,0.),
                          (array([[10,  5,  9,  4,  6,  4],
                                   [ 5,  2,  5,  2,  3,  6],
                                   [ 9,  5, 10,  2,  4,  1],
                                   [ 4,  2,  2,  2,  4,  6],
                                   [ 6,  3,  4,  4,  8,  7],
                                   [ 4,  6,  1,  6,  7,  2]]),
                            array([[ 1,  8,  9],
                                   [ 8, 10,  7],
                                   [ 9,  7,  1]]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.4,0.),
                          (stack([array([[10,  5,  9,  4,  6,  4],
                                         [ 5,  2,  5,  2,  3,  6],
                                         [ 9,  5, 10,  2,  4,  1],
                                         [ 4,  2,  2,  2,  4,  6],
                                         [ 6,  3,  4,  4,  8,  7],
                                         [ 4,  6,  1,  6,  7,  2]]),
                                  2*array([[10,  5,  9,  4,  6,  4],
                                           [ 5,  2,  5,  2,  3,  6],
                                           [ 9,  5, 10,  2,  4,  1],
                                           [ 4,  2,  2,  2,  4,  6],
                                           [ 6,  3,  4,  4,  8,  7],
                                           [ 4,  6,  1,  6,  7,  2]])]),
                           stack([array([[8, 2, 3],
                                         [2, 8, 5],
                                         [3, 5, 2]]),
                                  array([[ 1,  8,  9],
                                         [ 8, 10,  7],
                                         [ 9,  7,  1]])]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            2.4,0.)])

def test_anisotrop_heatexp_3d(cs,_as, xe, l, g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l
    xe = xe@(R@S).T
    #
    if len(cs.shape) == 2 and xe.shape[0]==1:
        Kes = fk_heatexp_aniso_3d(c=cs,a=_as,l=array([l,l,l]),g=array([g,g]))[None,:,:]
    elif len(cs.shape) == 2 and xe.shape[0]!=1:
        Kes = stack([fk_heatexp_aniso_3d(c=cs,a=_as,l=array([l,l,l]),g=array([g,g])) for i in range(xe.shape[0])])
    else:
        Kes = stack([fk_heatexp_aniso_3d(c=cs[i],a=_as[i],l=array([l,l,l]),g=array([g,g])) for i in range(cs.shape[0])])

    #
    assert_almost_equal(_fk_heatexp_3d(xe=xe,c=cs,a=_as),
                        Kes)
    return
