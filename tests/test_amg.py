from typing import Any, Callable, Dict, List, Tuple, Union

from numpy import array
from numpy.testing import assert_equal
from scipy.sparse import csc_array

from topoptlab.amg import rubestueben_coupling

import pytest

def test_rubestuebgen_coupling():

    test = array([[1., -0.25, 1., 0.55, 0.1],
                  [-0.25, 1., 0., 0., 0.],
                  [1., 0., 2., 1.2, 0.8],
                  [0.55, 0., 1.2, 5, -2.2],
                  [0.1, 0., 0.8, -2.2, 1.]])
    #
    solution = array([True, True, True, False, True, True, True,
                      True, False, True, True, False, False, True])
    #
    test = csc_array(test)
    diagonal = test.diagonal()
    test.setdiag(0)
    test.eliminate_zeros()
    # off diagonal maximum in each row
    max_row = test.power(2).sqrt().max(axis=1).todense()
    # extract indices and values
    i,j = test.nonzero()
    val = test[ i,j ]
    _,_,mask_strong = rubestueben_coupling(A=test)
    #
    assert_equal(solution, mask_strong)
    return
