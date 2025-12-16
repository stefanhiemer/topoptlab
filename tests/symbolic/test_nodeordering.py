from numpy import array
from numpy.testing import assert_almost_equal

from topoptlab.symbolic.utils import argsort_counterclock

import pytest


@pytest.mark.parametrize('coords,coords_sorted',
                         [(array([[ 1, -1],
                                  [ 1, 1],
                                  [-1, 1],
                                  [-1, -1]]),
                           array([[ 1, -1],
                                  [ 1, 1],
                                  [-1, 1],
                                  [-1, -1]])),
                          (array([[-1., -1., -1.],
                                  [ 1., -1., -1.],
                                  [ 1., 1., -1.],
                                  [-1., 1., -1.],
                                  [-1., -1., 1.],
                                  [ 1., -1., 1.],
                                  [ 1., 1., 1.],
                                  [-1., 1., 1.]]),
                           array([[-1., -1., -1.],
                                  [ 1., -1., -1.],
                                  [ 1., 1., -1.],
                                  [-1., 1., -1.],
                                  [-1., -1., 1.],
                                  [ 1., -1., 1.],
                                  [ 1., 1., 1.],
                                  [-1., 1., 1.]]))])

def test_conversion_consistency(coords,coords_sorted):
    
    assert_almost_equal(actual=coords,
                        desired=coords_sorted)
    return 
