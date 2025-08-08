from numpy import array
from numpy.testing import assert_equal,assert_array_equal
from scipy.sparse import csc_array

from topoptlab.amg import rubestueben_coupling

import pytest

def test_rubestuebgen_coupling():

    test = array([[1., -0.25, -1., 0.55, 0.1, 0.],
                  [-0.25, 1., 0., 0., 0., 0.],
                  [-1., 0., 2., -1.2, -0.1, 0.],
                  [0.55, 0., -1.2, 5, -2.2, 0.],
                  [0.1, 0., -0.1, -2.2, 1., 0], 
                  [0., 0., 0., 0., 0., 1.]])
    #
    solution_mask = array([True, True, True, False, 
                           True, 
                           True, True, False, 
                           False, True, True, 
                           False, False, True])
    
    solution_s = [[1,2,3],
                  [0],
                  [0,3],
                  [2,4],
                  [3],
                  []]
    solution_s_t = [[1,2],
                    [0],
                    [0,3],
                    [2,4],
                    [3], 
                    []]
    #
    test = csc_array(test)
    _,_,mask_strong,s,s_t = rubestueben_coupling(A=test, 
                                                 c_neg = 0.2, 
                                                 c_pos = 0.5)
    # for testing sort it
    s = [sorted(entry.tolist()) for entry in s]
    s_t = [sorted(entry.tolist()) for entry in s_t]
    
    #
    assert_equal(solution_mask, mask_strong)
    
    #
    for sol,actual in zip(solution_s,s):
        assert_equal(actual, sol)
    #
    for sol,actual in zip(solution_s_t,s_t):
        assert_equal(actual, sol)
