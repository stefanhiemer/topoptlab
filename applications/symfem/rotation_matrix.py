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
    print(R)
    # calculate derivatives
    if ndim == 1:
        pass 
    elif ndim > 1:
        R_dtheta = R.diff(theta)
        R_dtheta = simplify_matrix(R_dtheta)
    print(R_dtheta)
    if ndim > 2:
        R_dphi = R.diff(phi)
        R_dphi = simplify_matrix(R_dphi)
        print(R_dphi)