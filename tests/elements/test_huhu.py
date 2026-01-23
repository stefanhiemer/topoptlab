from numpy import array,loadtxt
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.huhu_2d import _lk_huhu_2d
from topoptlab.elements.huhu_3d import _lk_huhu_3d

@pytest.mark.parametrize('xe,ue',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2.1,-2],[2.1,-2],[2,2.1],[-2,2]]]), 
                           array([[0.,0.,
                                   0.,0.,
                                   0.1,0.2,
                                   0.,0.],
                                  [0.,0.,
                                   0.,0.,
                                   0.1,0.2,
                                   0.,0.]])), 
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                  [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]]), 
                          array([[0.,0.,0.,
                                  0.,0.,0.,
                                  0.1,0.2,0.3,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.],
                                 [0.,0.,0.,
                                  0.,0.,0.,
                                  0.1,0.2,0.3,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.,
                                  0.,0.,0.]]),) ])

def test_reproducible(xe,ue):
    
    ndim=xe.shape[-1]
    
    if ndim==2:
        k = _lk_huhu_2d(xe=xe,ue=ue,
                        exponent=1., 
                        kr=1.)
    elif ndim==3:
        k = _lk_huhu_3d(xe=xe,ue=ue,
                        exponent=1., 
                        kr=1.)
    #
    assert_allclose(k,
                    loadtxt(f"./tests/test_files/huhu{ndim}d.csv",
                                delimiter=",").reshape(k.shape))
    return
    