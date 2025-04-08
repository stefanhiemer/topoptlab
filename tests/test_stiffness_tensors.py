from itertools import combinations

from numpy import array
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.stiffness_tensors import compute_elastic_properties_3d

@pytest.mark.parametrize('',
                         [(),])

def test_compute_elastic_propertie_3d():
    E = 211
    K = 170
    lam = 3*K * (3*K-E) / (9*K-E)
    G = 3*K*E / (9*K-E)
    nu = (3*K-E) / (6*K)
    M = 3*K*(3*K+E) / (9*K-E)
    vals = array([E,nu,G,K,lam,M])
    keys = ["E","nu","G","K","lam","M"]
    for i,j in combinations(range(vals.shape[0]),2):
        elast_kws = {}
        elast_kws[keys[i]] = vals[i]
        elast_kws[keys[j]] = vals[j]
        assert_almost_equal(array(compute_elastic_properties_3d(**elast_kws)), 
                            vals) 
    return