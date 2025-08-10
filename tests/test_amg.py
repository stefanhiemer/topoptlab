from numpy import array
from numpy.testing import assert_equal,assert_array_equal
from scipy.sparse import csc_array

from topoptlab.amg import rubestueben_coupling

import pytest

@pytest.mark.parametrize('test, solution_mask, solution_s, solution_s_t',
                         [(array([[1., -0.25, -1., 0.55, 0.1, 0.],
                                  [-0.25, 1., 0., 0., 0., 0.],
                                  [-1., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., -0.1, -2.2, 1., 0], 
                                  [0., 0., 0., 0., 0., 1.]]), 
                           array([True, True, True, False, 
                                  True, 
                                  True, True, False, 
                                  False, True, True, 
                                  False, False, True]), 
                           [[1,2,3],
                            [0],
                            [0,3],
                            [2,4],
                            [3],
                            []], 
                           [[1,2],
                            [0],
                            [0,3],
                            [0,2,4],
                            [3], 
                            []]),
                          (array([[1., 0., -0.25, -1., 0.55, 0.1],
                                  [0., 1., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1],
                                  [0.55, 0., 0., -1.2, 5, -2.2],
                                  [0.1, 0., 0., -0.1, -2.2, 1.]]), 
                            array([True, True, True, False, 
                                   True, 
                                   True, True, False, 
                                   False, True, True, 
                                   False, False, True]), 
                            [[2,3,4],
                             [],
                             [0],
                             [0,4],
                             [3,5],
                             [4]], 
                            [[2,3],
                             [],
                             [0],
                             [0,4],
                             [0,3,5],
                             [4]]),
                          (array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1., 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]] ), 
                            array([True, True, True, False, 
                                   True, 
                                   True, True, False, 
                                   False, True, True, 
                                   False, False, True]), 
                            [[2,3,4],
                             [],
                             [0],
                             [0,4],
                             [3,5],
                             [4],
                             []], 
                            [[2,3],
                             [],
                             [0],
                             [0,4],
                             [0,3,5],
                             [4], 
                             []])])

def test_rubestuebgen_coupling(test, 
                               solution_mask, solution_s, solution_s_t):
    
    #
    test = csc_array(test)
    _,_,mask_strong,s,s_t = rubestueben_coupling(A=test, 
                                                 c_neg = 0.2, 
                                                 c_pos = 0.5)
    # for testing sort it
    s = [sorted(entry) for entry in s]
    s_t = [sorted(entry) for entry in s_t]
    
    #
    assert_equal(solution_mask, mask_strong)
    
    #
    for sol,actual in zip(solution_s,s):
        assert_equal(actual, sol)
    #
    for sol,actual in zip(solution_s_t,s_t):
        assert_equal(actual, sol)
