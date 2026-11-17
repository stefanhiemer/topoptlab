from topoptlab.symbolic.buckling import buckling_matrix
from topoptlab.symbolic.matrix_utils import generate_constMatrix
from topoptlab.symbolic.stiffness_tensors import stifftens_isotropic
from topoptlab.symbolic.code_conversion import convert_to_code

import pytest

@pytest.mark.parametrize('dim,sol',
                         [(1,
                           'np.array([[s[0]/l[0], -s[0]/l[0]],\n          [-s[0]/l[0], s[0]/l[0]]])')])

def test_bucklingmatrix_const(dim,sol):
    sigma = generate_constMatrix(ncol=1,nrow=int((dim**2 + dim) /2),
                                 name="s")
    code = convert_to_code(buckling_matrix(ndim=dim,
                                           sigma=sigma),
                          vectors=["l","g","s"],
                          npndarray=True,npcolumnstack=False)
    assert sol == code
    return

@pytest.mark.parametrize('dim,c,u,sol',
                         [(1, stifftens_isotropic(ndim=1),
                           generate_constMatrix(ncol=1,nrow=2,name="u"),
                           'np.array([[E*(-u[0] + u[1])/l[0]**2, E*(u[0] - u[1])/l[0]**2],\n          [E*(u[0] - u[1])/l[0]**2, E*(-u[0] + u[1])/l[0]**2]])'),
                          (1, None,
                           generate_constMatrix(ncol=1,nrow=2,name="u"),
                           'np.array([[c1*(-u[0] + u[1])/l[0]**2, c1*(u[0] - u[1])/l[0]**2],\n          [c1*(u[0] - u[1])/l[0]**2, c1*(-u[0] + u[1])/l[0]**2]])'),
                          (1, None, None,
                           'np.array([[c1*(-u[0] + u[1])/l[0]**2, c1*(u[0] - u[1])/l[0]**2],\n          [c1*(u[0] - u[1])/l[0]**2, c1*(-u[0] + u[1])/l[0]**2]])')])

def test_bucklingmatrix_disp(dim,c,u,sol):
    code = convert_to_code(buckling_matrix(ndim=dim,
                                           u=u,
                                           c=c),
                          matrices=["c"],vectors=["l","g","u"],
                          npndarray=True,npcolumnstack=False)
    assert sol == code
    return
