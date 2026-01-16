from sympy import symbols

import pytest

from topoptlab.symbolic.hyperelasticity import calculate_2pk, stvenant_engdensity, stvenant_2pk

@pytest.mark.parametrize('ndim',
                         [(1) ])

def test_stvenant_engdensity(ndim):
    #
    eng=stvenant_engdensity(ndim=ndim)
    #
    assert calculate_2pk(eng_density=eng,E=None,ndim=ndim)==stvenant_2pk(E=None,c=None,ndim=ndim)
    return 
