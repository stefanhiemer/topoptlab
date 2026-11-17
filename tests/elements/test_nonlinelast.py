from numpy import array,prod,ones
from numpy.random import seed,rand
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.material_models.stvenant import stvenant_matmodel
from topoptlab.material_models.neohooke import neohookean_matmodel
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.elements.nonlinear_elasticity import _lk_nonlinear_elast
from topoptlab.elements.check_tangents import check_tangent_fd

# distorted hexahedron
xe_3d = array([[[-1,-1,-1],[2,-1,-1],[1,2,-1],[-1,1,-1],
                [-1,-1,1],[1,-1,1.5],[1,1,1],[-1,1.5,1]]])
u_3d = array([[0.,0.,0.,0.,0.,0.,0.1,0.2,0.,0.,0.,0.,
               0.,0.,0.,0.,0.,0.3,0.2,0.1,0.,0.,0.,0.]])

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
                            "mu": ones((1,1))}),
                          (xe_3d,
                           u_3d,
                           stvenant_matmodel,
                           {"c": isotropic_3d()}),
                          (xe_3d,
                           u_3d,
                           neohookean_matmodel,
                           {"h": ones((1,1)),
                            "mu": ones((1,1))}),])

def test_material_models(xe,u,material_model,material_constants):
    #
    if len(xe.shape) == 2:
        xe = xe[None,...]
    #
    errs = check_tangent_fd(Ke_fe = _lk_nonlinear_elast,
                            Ke_fe_args = {"xe": xe,
                                          "material_model": material_model,
                                          "material_constants": material_constants},
                            u = u)
    assert errs.max() < 1e-7
    return
