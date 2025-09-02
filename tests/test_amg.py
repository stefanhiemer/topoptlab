from numpy import array
from numpy.testing import assert_equal,assert_allclose
from scipy.sparse import csc_array

from topoptlab.amg import rubestueben_coupling

import pytest

@pytest.mark.parametrize('test, sol_mask, sol_s, sol_s_t, sol_iso',
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
                            []],
                           [5]),
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
                             [4]],
                            [1]),
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
                             []],
                            [1,6])])

def test_rubestuebgen_coupling(test, 
                               sol_mask, sol_s, sol_s_t, sol_iso):
    
    #
    test = csc_array(test)
    _,_,mask_strong,s,s_t,iso = rubestueben_coupling(A=test, 
                                                     c_neg = 0.2, 
                                                     c_pos = 0.5)
    # for testing sort it
    s = [sorted(entry) for entry in s]
    s_t = [sorted(entry) for entry in s_t]
    #
    assert_equal(sol_mask, mask_strong)
    #
    for sol,actual in zip(sol_s,s):
        assert_equal(actual, sol)
    #
    for sol,actual in zip(sol_s_t,s_t):
        assert_equal(actual, sol)
    #
    assert_equal(sol_iso, iso)
    return

from topoptlab.amg import standard_coarsening

@pytest.mark.parametrize('test, sol_mask',
                         [(array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1., 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]]), 
                          array([False, False, True, False, True, False, False]))])

def test_standard_coarsening(test, sol_mask):
    #
    test = csc_array(test)
    #
    mask_coarse = standard_coarsening(test,
                                      coupling_fnc=rubestueben_coupling,
                                      coupling_kw = {"c_neg": 0.2, 
                                                     "c_pos": 0.5})
    #
    assert_equal(sol_mask, mask_coarse)
    return

from topoptlab.amg import direct_interpolation

@pytest.mark.parametrize('test, solution',
                         [(array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1.7, 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]] ), 
                          array( [[1.25, -0.65],
                                  [0., 0.],
                                  [1., 0.],
                                  [0., 1.15],
                                  [0., 1.],
                                  [0., 1.35294118],
                                  [0., 0.]]))])

def test_direct_interpolation(test, solution):
    
    
    #
    test = csc_array(test)
    #
    mask_coarse = standard_coarsening(test,
                                      coupling_fnc=rubestueben_coupling,
                                      coupling_kw = {"c_neg": 0.2, 
                                                     "c_pos": 0.5})
    #
    P = direct_interpolation(test, mask_coarse)
    #
    assert_allclose(P.toarray(),solution)
    return
