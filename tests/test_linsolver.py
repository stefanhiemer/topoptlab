from functools import partial

from numpy.testing import assert_almost_equal
from numpy import array
from scipy.sparse import csc_matrix,eye
import pytest

from topoptlab.linear_solvers import gauss_seidel ,smoothed_jacobi,\
    modified_richardson, successive_overrelaxation,pcg,cg
@pytest.mark.parametrize('fun',
                         [(gauss_seidel),
                          (smoothed_jacobi),
                          (modified_richardson),
                          (successive_overrelaxation),
                          ( partial(pcg,P=eye(3,3)) ),
                          cg])

def test_linsystem(fun):
    
    #
    A = csc_matrix([[2,1,0],[1,2,0],[0,0,1]])
    #
    b = array([1,2,0])
    # solution
    sol = array([0., 1., 0.])
    #
    assert_almost_equal(fun(A,b)[0], sol)
    return