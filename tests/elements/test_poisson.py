from numpy import array,stack,eye,vstack,tan,pi
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.poisson_2d import _lk_poisson_2d, lk_poisson_2d

@pytest.mark.parametrize('ks, xe, l, g',
                         [([1.],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2,0.),
                          ([1.,2.],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.2,0.),
                          ([1.,2.],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.3,0.),])

def test_isotrop_poisson_2d(ks,xe,l,g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l/2
    xe = xe@(R@S).T
    #
    Kes = stack([lk_poisson_2d(k=k,l=array([l,l]),g=array([g])) for k in zip(ks)])
    #
    ks = array(ks)[:,None,None]*eye(2)[None,:,:]
    assert_almost_equal(_lk_poisson_2d(xe=xe,k=ks),
                        Kes)
    return


from topoptlab.elements.poisson_2d import lk_poisson_aniso_2d

@pytest.mark.parametrize('ks, xe, l, g',
                         [(array([[1,2],[2,1]]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2,0.),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1.2,0.),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           2.4,0.),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]),
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           3.2,0.),])

def test_anisotrop_poisson_2d(ks,xe,l,g):
    # affine deform box
    R = eye(2)
    R[0,1] = tan(g)
    S = eye(2)*l/2
    xe = xe@(R@S).T
    #
    if len(ks.shape) == 2:
        Kes = lk_poisson_aniso_2d(k=ks,l=array([l,l]),g=array([g]))[None,:,:]
    else:
        Kes = stack([lk_poisson_aniso_2d(k=ks[i],l=array([l,l]),g=array([g])) \
                     for i in range(ks.shape[0])])
    #
    assert_almost_equal(_lk_poisson_2d(xe,k=ks),
                        Kes)
    return

from topoptlab.elements.poisson_3d import _lk_poisson_3d,lk_poisson_3d

@pytest.mark.parametrize('ks, xe, l, g',
                         [([1.],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            1.2,0.),
                          ([1.5,2],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            1.2,0.),
                          ([1.3,2],
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            2,0.)])

def test_isotrop_poisson_3d(ks,xe,l,g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l/2
    xe = xe@(R@S).T
    #
    Kes = stack([lk_poisson_3d(k=k,l=array([l,l,l]),g=array([g,g])) for k in zip(ks)])
    #
    ks = array(ks)[:,None,None]*eye(3)[None,:,:]
    #
    assert_almost_equal(_lk_poisson_3d(xe,k=ks),
                        Kes)
    return

from topoptlab.elements.poisson_3d import lk_poisson_aniso_3d

@pytest.mark.parametrize('ks, xe, l, g',
                         [(array([[1,2,0],[2,1,0],[0,0,1]]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            1.2,1.),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            2.,pi/4),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]),
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1.5,pi/3)])

def test_anisotrop_poisson_3d(ks,xe,l,g):
    # affine deform box
    R = eye(3)
    R[0,1] = tan(g)
    R[0,2] = tan(g)
    S = eye(3)*l/2
    xe = xe@(R@S).T
    #
    if len(ks.shape) == 2:
        Kes = lk_poisson_aniso_3d(k=ks,l=array([l,l,l]),g=array([g,g]))[None,:,:]
    else:
        Kes = stack([lk_poisson_aniso_3d(k=ks[i],l=array([l,l,l]),g=array([g,g])) \
                     for i in range(ks.shape[0])])
    #
    assert_almost_equal(_lk_poisson_3d(xe,k=ks),
                        Kes)
    return

@pytest.mark.parametrize('xe',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]])),
                          array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                  [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]])])

def test_consist(xe):
    ndim = xe.shape[-1]
    k = stack([ eye(ndim)*(i+1) for i in range(xe.shape[0])] )
    if ndim == 2:
        #
        Kes = vstack([_lk_poisson_2d(xe[i],k=k[i]) for i in range(xe.shape[0])])
        #
        assert_almost_equal(_lk_poisson_2d(xe,k=k),
                            Kes)
    elif ndim == 3:
        #
        Kes = vstack([_lk_poisson_3d(xe[i],k=k[i]) for i in range(xe.shape[0])])
        #
        assert_almost_equal(_lk_poisson_3d(xe,k=k),
                            Kes)
    return
