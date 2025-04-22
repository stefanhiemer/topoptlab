from numpy import array,stack,vstack
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.bodyforce_2d import _lf_bodyforce_2d,lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import _lf_bodyforce_3d, lf_bodyforce_3d

@pytest.mark.parametrize('xe',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                    [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])])

def test_compareanalyt(xe):
    
    l = (xe.max(axis=1)-xe.min(axis=1))[0]/2
    if xe.shape[-1] == 2:
        #
        fes = stack([lf_bodyforce_2d(l=l) for i in range(xe.shape[0])])
        #
        assert_allclose(_lf_bodyforce_2d(xe=xe),
                        fes)
    elif xe.shape[-1] == 3:
        #
        fes = stack([lf_bodyforce_3d(l=l) for i in range(xe.shape[0])])
        #
        assert_allclose(_lf_bodyforce_3d(xe=xe),
                        fes)
    return

@pytest.mark.parametrize('xe',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-2,-2.1],[2.1,-2],[2,2],[-2,2]]])),
                          array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-2.1,-2,-2],[2.1,-2,-2],[2,2.1,-2],[-2,2,-2],
                                  [-2,-2,2],[2,-2,2.1],[2,2,2],[-2,2.1,2]]])])

def test_consist(xe):
    
    if xe.shape[-1] == 2:
        #
        fes = vstack([_lf_bodyforce_2d(xe[i]) for i in range(xe.shape[0])])
        #
        assert_allclose(_lf_bodyforce_2d(xe),
                        fes)
    elif xe.shape[-1] == 3:
        #
        fes = vstack([_lf_bodyforce_3d(xe[i]) for i in range(xe.shape[0])])
        #
        assert_allclose(_lf_bodyforce_3d(xe),
                        fes)
    return
    