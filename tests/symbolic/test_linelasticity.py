from topoptlab.symbolic.lin_elasticity import stiffness_matrix
from topoptlab.symbolic.stiffness_tensors import stifftens_isotropic
from topoptlab.symbolic.code_conversion import convert_to_code 

import pytest

@pytest.mark.parametrize('dim,c,sol',
                         [(1, stifftens_isotropic(ndim=1),
                           'np.array([[E/l[0], -E/l[0]],\n          [-E/l[0], E/l[0]]])'),
                          (1, None,
                            'np.array([[c1/l[0], -c1/l[0]],\n          [-c1/l[0], c1/l[0]]])')])

def test_stiffnessmatrix(dim,c,sol):
    code = convert_to_code(stiffness_matrix(c=c, 
                                           plane_stress=True, ndim=dim),
                          matrices=["c"],vectors=["l","g"])
    assert sol == code
    return 

from topoptlab.symbolic.lin_elasticity import strainforces

@pytest.mark.parametrize('dim,c,sol',
                         [(1, stifftens_isotropic(ndim=1),
                           'np.array([[-E*eps[0]],\n          [E*eps[0]]])'),
                          (1, None,
                            'np.array([[-c1*eps[0]],\n          [c1*eps[0]]])')])

def test_strainforces(dim,c,sol):
    code = convert_to_code(strainforces(c=c, 
                                       plane_stress=True, ndim=dim),
                          matrices=["c"],vectors=["l","g", "eps"])
    assert sol == code
    return 
