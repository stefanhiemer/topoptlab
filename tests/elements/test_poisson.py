from numpy import array,stack,eye,vstack
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.poisson_2d import _lk_poisson_2d, lk_poisson_2d

@pytest.mark.parametrize('ks, xe',
                         [([1.], 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          ([1.,2.],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          ([1.,2.],
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),])

def test_isotrop_poisson_2d(ks,xe):
    
    #
    Kes = stack([lk_poisson_2d(k) for k in zip(ks)])
    #
    ks = array(ks)[:,None,None]*eye(2)[None,:,:]
    assert_almost_equal(_lk_poisson_2d(xe=xe,k=ks),
                        Kes)
    return


from topoptlab.elements.poisson_2d import lk_poisson_aniso_2d

@pytest.mark.parametrize('ks, xe',
                         [(array([[1,2],[2,1]]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (stack([array([[1,2],[2,1]]),array([[2,3],[3,2]])]), 
                           array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),])

def test_anisotrop_poisson_2d(ks,xe):
    
    #
    if len(ks.shape) == 2:
        Kes = lk_poisson_aniso_2d(ks)[None,:,:]
    else:
        Kes = stack([lk_poisson_aniso_2d(ks[i]) for i in range(ks.shape[0])])
    #
    assert_almost_equal(_lk_poisson_2d(xe,k=ks),
                        Kes)
    return

from topoptlab.elements.poisson_3d import _lk_poisson_3d,lk_poisson_3d

@pytest.mark.parametrize('ks, xe',
                         [([1.], 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          ([1,2], 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          ([1,2], 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))])

def test_isotrop_poisson_3d(ks,xe):
    
    #
    Kes = stack([lk_poisson_3d(k) for k in zip(ks)])
    #
    ks = array(ks)[:,None,None]*eye(3)[None,:,:]
    #
    assert_almost_equal(_lk_poisson_3d(xe,k=ks),
                        Kes)
    return

from topoptlab.elements.poisson_3d import lk_poisson_aniso_3d

@pytest.mark.parametrize('ks, xe',
                         [(array([[1,2,0],[2,1,0],[0,0,1]]), 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]), 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          (stack([array([[1,2,0],[2,1,0],[0,0,1]]),
                                  array([[1,2,3],[2,1,0],[3,0,1]])]), 
                           array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                  [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                   [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]))])

def test_anisotrop_poisson_3d(ks,xe):
    
    #
    if len(ks.shape) == 2:
        Kes = lk_poisson_aniso_3d(ks)[None,:,:]
    else:
        Kes = stack([lk_poisson_aniso_3d(ks[i]) for i in range(ks.shape[0])])
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