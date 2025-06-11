from numpy import array,stack,vstack
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.mass_scalar_2d import _lm_mass_2d,lm_mass_2d
from topoptlab.elements.mass_scalar_3d import _lm_mass_3d, lm_mass_3d

@pytest.mark.parametrize('xe',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])])

def test_compareanalyt(xe):
    
    l = (xe.max(axis=1)-xe.min(axis=1))[0]
    if xe.shape[-1] == 2:
        #
        Kes = stack([lm_mass_2d(l=l) for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_2d(xe=xe),
                        Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = stack([lm_mass_3d(l=l) for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_3d(xe=xe),
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
    
    if xe.shape[-1] == 2:
        #
        Kes = vstack([_lm_mass_2d(xe[i]) for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_2d(xe),
                        Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = vstack([_lm_mass_3d(xe[i]) for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_3d(xe),
                        Kes)
    return
    