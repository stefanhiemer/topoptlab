# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.symbols import x
from symfem.functions import MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, simplify_matrix
from topoptlab.symbolic.voigt import convert_from_voigt
from topoptlab.symbolic.code_conversion import convert_to_code 
from topoptlab.symbolic.parametric_map import jacobian

if __name__ == "__main__":
    #
    ndim = 2
    # general stiffness tensor
    size = int((ndim**2 + ndim) /2)
    _c = generate_constMatrix(nrow=size, ncol=size,
                              name="c",symmetric=True)
    # reduce to orthotropic
    c = [ [0 for j in range(size)] for i in range(size) ]
    print(c)
    for i in range(ndim):
        for j in range(ndim):
            c[i][j] = _c[i,j]
            c[j][i] = _c[i,j]
    for i in range(ndim,size):
        c[i][i] = _c[i,i]
    print(c)
    c = MatrixFunction(c)
    # general strain tensor in Voigt notation
    strain = generate_constMatrix(nrow=size, ncol=1, name="E")
    # get matrix represenation of stress
    stress = convert_from_voigt(A_v=c@strain).as_sympy()
    print(stress)
    #
    eigenvectors = stress.eigenvects()
    invariants = stress.eigenvals()
    for i,eigval in enumerate(invariants.keys()):
        print("invariant ",i)
        print(eigval)
