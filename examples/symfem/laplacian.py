# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols

from topoptlab.symbolic.matrix_utils import diag,generate_constMatrix
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.operators import aniso_laplacian,nonlin_laplacian

if __name__ == "__main__":
    
    # isotropic heat conduction
    for dim in range(1,4):
        k = diag( [symbols("k") for i in range(dim)] )
        print(str(dim)+"D")
        print(convert_to_code(aniso_laplacian(ndim = dim, K=k),
                              matrices=["k"],vectors=["l","g"]),"\n")

    # general anisotropic heat conduction
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(aniso_laplacian(ndim = dim),
                              matrices=["k"],vectors=["l","g"]),"\n")
    import sys 
    sys.exit()
    #
    ndim = 2
    K = generate_constMatrix(ncol=ndim,nrow=ndim,
                             name="a",
                             symmetric=True) + \
        generate_constMatrix(ncol=ndim,nrow=ndim,
                                 name="b",
                                 symmetric=True).__mul__(symbols("phi"))
    print(K,"\n")
    print(nonlin_laplacian(ndim=ndim,
                           K=K,
                           linearization="picard"))
    
