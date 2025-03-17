import math 

from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import rotation_matrix, generate_constMatrix
from topoptlab.symfem_utils import simplify_matrix,convert_to_code
from topoptlab.symfem_utils import convert_to_voigt

if __name__ == "__main__":
    #
    ndim=3
    #
    R = rotation_matrix(ndim=ndim)
    #
    if ndim == 1:
        pass 
    elif ndim == 2:
        theta = next(iter(R[0,0].as_sympy().free_symbols))
    elif ndim == 3:
        #
        symb = R[0,0].as_sympy().free_symbols
        # sort to always get same result
        symb = list(symb)
        inds = sorted(range(len(symb)), key=[str(s) for s in symb].__getitem__)
        symb = [symb[i] for i in inds]
        #
        phi,theta = symb
        # print just to check
        print(phi,theta)
    # abstract tensor to be rotated
    A = generate_constMatrix(ncol=ndim, 
                             nrow=ndim, 
                             name="A",
                             symmetric=True)
    # rotate tensor
    A_rotated = R.transpose()@A@R
    # rewrite in Voigt notation
    A_rotated = convert_to_voigt(A_rotated)
    A_rotated = simplify_matrix(A_rotated)
    # number of entries of the Voigt vector
    nv = A_rotated.shape[0]
    # get symbols of A to do a factorization
    A_symbols = convert_to_voigt(A)
    A_symbols = MatrixFunction([[next(iter(A_symbols[i,0].as_sympy().free_symbols))] \
                                for i in range(nv)])
    # factorize according to entries of A in Voigt order which gives the rotation matrix 
    # in vogit notation
    Rv = [[A_rotated[j,0].as_sympy().coeff(A_symbols[i,0]) for j in range(nv)] \
              for i in range(nv)]
    Rv = MatrixFunction(Rv)
    Rv = simplify_matrix(Rv)
    print(convert_to_code(Rv,npndarray=False,
                          max_line_length=100))
    # calculate derivatives
    if ndim == 1:
        pass 
    elif ndim > 1:
        Rv_dtheta = Rv.diff(theta)
        Rv_dtheta = simplify_matrix(Rv_dtheta)
    print(Rv_dtheta)
    if ndim > 2:
        Rv_dphi = Rv.diff(phi)
        Rv_dphi = simplify_matrix(Rv_dphi)
        print(Rv_dphi)