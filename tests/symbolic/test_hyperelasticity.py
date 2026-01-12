from sympy import symbols

import pytest

from topoptlab.symbolic.hyperelasticity import calculate_2PK, stvenant_engdensity, stvenant_2PK

@pytest.mark.parametrize('ndim',
                         [(1) ])

def test_stvenant_engdensity(ndim):
    #
    eng=stvenant_engdensity(ndim=ndim)
    #
    assert calculate_2PK(eng_density=eng,E_v=None,ndim=ndim)==stvenant_2PK(E_v=None,c=None,ndim=ndim)
    return 
