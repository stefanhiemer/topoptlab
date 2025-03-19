from numpy import arange,array,pi
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.rotation_matrices import Rv_2d

@pytest.mark.parametrize('c, R, c_ref',
                         [(array([[1., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]),
                           Rv_2d(array([pi/2]))[0,:,:],
                           array([[0., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 0.]])), 
                          (array([[2., 1., 0.],
                                  [1., 1., 0.],
                                  [0., 0., 1.]]),
                           Rv_2d(array([pi/2]))[0,:,:],
                           array([[1., 1., 0.],
                                   [1., 2., 0.],
                                   [0., 0., 1.]])), 
                          (array([[1., 1., 0.],
                                  [1., 2., 0.],
                                  [0., 0., 1.]]),
                           Rv_2d(array([pi/2]))[0,:,:],
                           array([[2., 1., 0.],
                                   [1., 1., 0.],
                                   [0., 0., 1.]])), 
                          (array([[1., 1., 0.],
                                  [1., 2., 0.],
                                  [0., 0., 1.]]),
                           Rv_2d(array([pi]))[0,:,:],
                           array([[1., 1., 0.],
                                  [1., 2., 0.],
                                  [0., 0., 1.]]))],
                         )

def test_rotmatrix_voigt_2D(c,R,c_ref):
    
    assert_almost_equal( R.T@c@R, c_ref)
    
    return