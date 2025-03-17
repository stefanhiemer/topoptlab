import math 

from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import rotation_matrix, generate_constMatrix
from topoptlab.symfem_utils import simplify_matrix,convert_to_code
from topoptlab.symfem_utils import convert_to_voigt,rotation_matrix_dangle

if __name__ == "__main__":
    #
    ndim=2
    #
    Rv = rotation_matrix(ndim=ndim,mode="voigt")
    # get angle symbolic variables
    if ndim == 1:
        pass 
    elif ndim == 2:
        theta = next(iter(Rv[0,0].as_sympy().free_symbols))
    elif ndim == 3:
        #
        symb = Rv[0,0].as_sympy().free_symbols
        # sort to always get same result
        symb = list(symb)
        inds = sorted(range(len(symb)), key=[str(s) for s in symb].__getitem__)
        symb = [symb[i] for i in inds]
        #
        phi,theta = symb
        # print just to check
        print(phi,theta)
    # stiffness tensor
    c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2), 
                             nrow=int((ndim**2 + ndim) /2), 
                             name="c",
                             symmetric=True)
    c_rot = Rv.transpose()@c@Rv
    c_rot = simplify_matrix(c_rot)
    #print(c_rot)
    #print()
    if ndim == 1:
        pass 
    elif ndim > 1:
        c_rot_dtheta = c_rot.diff(theta)
        c_rot_dtheta = simplify_matrix(c_rot_dtheta)
    #print(c_rot_dtheta)
    if ndim > 2:
        c_rot_dphi = c_rot.diff(phi)
        c_rot_dphi = simplify_matrix(c_rot_dphi)
        #print()
        #print(c_rot_dphi)
    #
    if ndim == 2:
        dRvdtheta = rotation_matrix_dangle(ndim,mode="voigt")
        _c_rot_dtheta = dRvdtheta.transpose()@c@Rv + Rv.transpose()@c@dRvdtheta
        _c_rot_dtheta = simplify_matrix(_c_rot_dtheta)
        print(c_rot_dtheta == _c_rot_dtheta)
    elif ndim == 3:
        dRvdtheta,dRvdphi = rotation_matrix_dangle(ndim,mode="voigt")
        _c_rot_dtheta = dRvdtheta.transpose()@c@Rv + Rv.transpose()@c@dRvdtheta
        _c_rot_dtheta = simplify_matrix(_c_rot_dtheta)
        _c_rot_dphi = dRvdphi.transpose()@c@Rv + Rv.transpose()@c@dRvdphi
        _c_rot_dphi = simplify_matrix(_c_rot_dphi)
        print(c_rot_dtheta == _c_rot_dtheta)
        print(c_rot_dphi == _c_rot_dphi)