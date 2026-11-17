from numpy import array, ones
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.linear_interval import shape_functions_dxi, jacobian,\
                                               invjacobian, bmatrix
from topoptlab.elements.poisson import _lk_poisson
from topoptlab.elements.linear_elasticity import _lk_linear_elast, _lf_strain
from topoptlab.elements.mass_scalar import _lm_mass as _lm_mass_scalar
from topoptlab.elements.mass_vector import _lm_mass as _lm_mass_vector
from topoptlab.elements.bodyforce import _lf_bodyforce
from topoptlab.elements.monomial_scalar import _lm_monomial
from topoptlab.elements.advection import _lk_advection
from topoptlab.elements.advection_1d import lk_advection_1d
from topoptlab.elements.intertia import lk_intertia
from topoptlab.elements.inertia_1d import lk_inertia_1d, lf_convective_1d
from topoptlab.elements.nonlinear_elasticity import _lk_nonlinear_elast
from topoptlab.elements.bar_1d import lk_bar_1d
from topoptlab.elements.check_tangents import check_tangent_fd
from topoptlab.material_models.stvenant import stvenant_matmodel
from topoptlab.material_models.neohooke import neohookean_matmodel

# two elements of length 3 and 2
xe = array([[[1.],[4.]],
            [[0.],[2.]]])
l = array([3.,2.])
# cross section areas
A = array([0.5,2.])

def test_mapping():
    xi = array([-0.5,0.5])
    assert shape_functions_dxi(xi).shape == (2,2,1)
    assert shape_functions_dxi(0.).shape == (1,2,1)
    #
    J = jacobian(xi, xe=xe, all_elems=True)
    assert J.shape == (4,1,1)
    assert_allclose(J[:,0,0], [1.5,1.5,1.,1.])
    #
    Jinv, detJ = invjacobian(xi, xe=xe, all_elems=True, return_det=True)
    assert Jinv.shape == (4,1,1)
    assert_allclose(Jinv[:,0,0]*detJ, 1.)
    #
    B, detJ = bmatrix(xi=xi, xe=xe, all_elems=True, return_detJ=True)
    assert B.shape == (4,1,2)
    assert_allclose(B[:,0,1], 1/array([3.,3.,2.,2.]))
    assert_allclose(B.sum(axis=-1), 0.)
    return

@pytest.mark.parametrize('k', [1., 2.5])
def test_poisson_elast(k):
    ref = array([lk_bar_1d(E=k, A=_A, l=_l)[0] for _A,_l in zip(A,l)])
    assert_allclose(_lk_poisson(xe=xe, k=array([[k]]), t=A), ref)
    assert_allclose(_lk_linear_elast(xe=xe, c=array([[k]]), t=A), ref)
    # uniform strain
    assert_allclose(_lf_strain(xe=xe, eps=array([0.1]), c=array([[k]]), t=A)[:,:,0],
                    array([[-1.,1.],[-1.,1.]])*k*0.1*A[:,None])
    return

def test_mass_bodyforce():
    ref = array([_l/6*array([[2.,1.],[1.,2.]]) for _l in l])
    assert_allclose(_lm_mass_scalar(xe=xe, t=A), A[:,None,None]*ref)
    assert_allclose(_lm_mass_vector(xe=xe, t=A), A[:,None,None]*ref)
    assert_allclose(_lm_monomial(xe=xe, u=ones((2,2)), n=0), ref)
    assert_allclose(_lf_bodyforce(xe=xe, b=array([2.]), t=A)[:,:,0],
                    array([[_A*_l,_A*_l] for _A,_l in zip(A,l)]))
    # scalar thickness
    assert_allclose(_lm_mass_scalar(xe=xe, t=0.5), 0.5*ref)
    return

@pytest.mark.parametrize('v', [array([1.,2.]), array([-0.5,3.])])
def test_advection_inertia(v):
    for i in range(xe.shape[0]):
        assert_allclose(_lk_advection(xe=xe[i], ve=v[None]),
                        lk_advection_1d(v=v[:,None], l=l[i:i+1]))
        K, f = lk_intertia(xe=xe[i], ve=v[None])
        assert_allclose(K, lk_inertia_1d(v=v[:,None], l=l[i:i+1]))
        assert_allclose(f.ravel(), lf_convective_1d(v=v[:,None], l=l[i:i+1]).ravel())
    return

@pytest.mark.parametrize('material_model, material_constants',
                         [(stvenant_matmodel, {"c": array([[2.]])}),
                          (neohookean_matmodel, {"h": ones((2,1)),
                                                 "mu": ones((2,1))})])
def test_nonlinear(material_model, material_constants):
    errs = check_tangent_fd(Ke_fe = _lk_nonlinear_elast,
                            Ke_fe_args = {"xe": xe,
                                          "material_model": material_model,
                                          "material_constants": material_constants,
                                          "t": A},
                            u = array([[0.,0.3],[0.1,-0.2]]),
                            verbose=False)
    assert errs.max() < 1e-7
    # cross section scales tangent and internal force
    u = array([[0.,0.3],[0.1,-0.2]])
    K1, f1 = _lk_nonlinear_elast(xe=xe, ue=u, material_model=material_model,
                                 material_constants=material_constants)
    KA, fA = _lk_nonlinear_elast(xe=xe, ue=u, material_model=material_model,
                                 material_constants=material_constants, t=A)
    assert_allclose(KA, A[:,None,None]*K1)
    assert_allclose(fA, A[:,None]*f1)
    return
