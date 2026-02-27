from numpy import array,prod,ones
from numpy.random import seed,rand
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.material_models.stvenant import stvenant_matmodel
from topoptlab.material_models.neohooke import neohookean_matmodel
from topoptlab.stiffness_tensors import isotropic_2d
from topoptlab.elements.nonlinear_elasticity_2d import _lk_nonlinear_elast_2d
from topoptlab.elements.check_tangents import check_tangent_fd

@pytest.mark.parametrize('xe, u, material_model, material_constants',
                         [(array([[[-1,-1],[2,-1],[1,2],[-1,1]]]),
                           array([[0.,0.,0.,0.,1.,1.,0.,0.]]),
                           stvenant_matmodel,
                           {"c": ones((3,3))}),
                          (array([[[-1,-1],[2,-1],[1,2],[-1,1]]]),
                            array([[0.,0.,0.,0.,1.,1.,0.,0.]]),
                            stvenant_matmodel,
                            {"c": isotropic_2d()}),
                          (array([[[-1,-1],[2,-1],[1,2],[-1,1]]]),
                           array([[0.,0.,0.,0.,1.,1.,0.,0.]]),
                           neohookean_matmodel,
                           {"h": ones((1,1)),
                            "mu": ones((1,1))}),])

def test_material_models(xe,u,material_model,material_constants):
    #
    if len(xe.shape) == 2:
        xe = xe[None,...]
    #
    errs = check_tangent_fd(Ke_fe = _lk_nonlinear_elast_2d,
                            Ke_fe_args = {"xe": xe,
                                          "material_model": material_model,
                                          "material_constants": {"c": ones((3,3)), 
                                                                 "h": ones((xe.shape[0],1)),
                                                                 "mu": ones((xe.shape[0],1))}},
                            u = u)
    assert errs.max() < 1e-7
    return