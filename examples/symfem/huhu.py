# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols

from topoptlab.symbolic.matrix_utils import diag,generate_constMatrix
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.tmc import huhu_engdensity

if __name__ == "__main__":
    #
    print("Squared Hessian")
    for dim in range(3,4):
        k = diag( [symbols("k") for i in range(dim)] )
        print(str(dim)+"D")
        print(huhu_engdensity(u=None,ndim=dim))
        #print(convert_to_code(huhu(ndim.ndim=dim),
        #                      matrices=[],vectors=["l","g"]),"\n")
