from numpy import array,stack,eye
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.mass_2d import _lm_mass_2d,lm_mass_symfem
from topoptlab.elements.mass_3d import _lm_mass_3d, lm_mass_3d

@pytest.mark.parametrize('xe',
                         [(array([[[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                                  [[-1,-1],[1,-1],[1,1],[-1,1]]])),
                          (array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])),
                          array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                                 [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])])

def test_massmatrix(xe):
    
    if xe.shape[-1] == 2:
        #
        Kes = stack([lm_mass_symfem() for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_2d(xe),
                        Kes)
    elif xe.shape[-1] == 3:
        #
        Kes = stack([lm_mass_3d() for i in range(xe.shape[0])])
        #
        assert_allclose(_lm_mass_3d(xe),
                        Kes)
    return
    