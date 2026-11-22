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
from numpy import eye, ix_, fill_diagonal, full, zeros
from numpy.linalg import inv
from sympy import symbols

from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.symbolic.stiffness_tensors import stifftens_isotropic

def _isotropic_reference(E, nu, ndim, plane_stress=True):
    """
    Isotropic stiffness tensor (engineering Voigt notation) from the
    textbook compliance tensor, independent of the implementation. Plane
    strain is the in-plane part of the 3D tensor, plane stress the inverse of
    the in-plane part of the 3D compliance.
    """
    S = zeros((6,6))
    S[:3,:3] = full((3,3), -nu/E)
    fill_diagonal(S[:3,:3], 1/E)
    S[3:,3:] = eye(3)*2*(1+nu)/E
    if ndim == 3:
        return inv(S)
    # in-plane components xx, yy, xy
    inds = [0,1,5]
    if plane_stress:
        return inv(S[ix_(inds,inds)])
    else:
        return inv(S)[ix_(inds,inds)]

@pytest.mark.parametrize('E, nu', [(1., 0.3), (211., 0.27), (2.5, 0.45)])

def test_isotropic(E, nu):
    # numerical tensors
    assert_almost_equal(isotropic_2d(E=E, nu=nu, plane_stress=True),
                        _isotropic_reference(E, nu, 2, plane_stress=True))
    assert_almost_equal(isotropic_2d(E=E, nu=nu, plane_stress=False),
                        _isotropic_reference(E, nu, 2, plane_stress=False))
    assert_almost_equal(isotropic_3d(E=E, nu=nu),
                        _isotropic_reference(E, nu, 3))
    # symbolic tensors
    _E, _nu = symbols("E nu")
    for ndim, plane_stress in [(2, True), (2, False), (3, True)]:
        c = stifftens_isotropic(ndim=ndim, plane_stress=plane_stress)
        c = array(c.as_sympy().subs({_E: E, _nu: nu}), dtype=float)
        assert_almost_equal(c, _isotropic_reference(E, nu, ndim,
                                                    plane_stress=plane_stress))
    return
