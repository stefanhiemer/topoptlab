from sympy import symbols

from topoptlab.symbolic.matrix_utils import diag
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.operators import aniso_laplacian


import pytest


@pytest.mark.parametrize('ndim,sol',
                         [(1,
                           'np.array([[k/l[0], -k/l[0]],\n          [-k/l[0], k/l[0]]])') ])

def test_anistropiclaplacian(ndim,sol):
    #
    k = diag( [symbols("k") for i in range(ndim)] )
    code = convert_to_code(aniso_laplacian(ndim = ndim, K=k),
                           matrices=["k"],vectors=["l","g"])
    assert sol == code
    return 
