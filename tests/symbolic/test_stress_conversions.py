from sympy import symbols
 
from topoptlab.symbolic.matrix_utils import generate_constMatrix,matrix_equal
from topoptlab.symbolic.stress_conversions import cauchy_to_pk1,pk1_to_cauchy
from topoptlab.symbolic.stress_conversions import cauchy_to_pk2,pk2_to_cauchy
from topoptlab.symbolic.stress_conversions import pk1_to_pk2,pk2_to_pk1

import pytest


@pytest.mark.parametrize('forward,backward,ndim',
                         [(cauchy_to_pk1,pk1_to_cauchy,2),
                          (cauchy_to_pk1,pk1_to_cauchy,3),
                          (cauchy_to_pk2,pk2_to_cauchy,2),
                          (cauchy_to_pk2,pk2_to_cauchy,3),
                          (pk2_to_pk1,pk1_to_pk2,2),
                          (pk2_to_pk1,pk1_to_pk2,3)])

def test_conversion_consistency(forward,backward,ndim):
    # def. grad.
    F = generate_constMatrix(ncol=ndim,nrow=ndim,
                             name="F",symmetric=False)
    # arbitrary stress
    T = generate_constMatrix(ncol=ndim,nrow=ndim,
                             name="T",symmetric=False)
    assert matrix_equal( T, backward(forward(T,F=F),F=F) )
    return 
