from symfem.functions import VectorFunction, MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell,scale_cell
from topoptlab.symfem_utils import convert_to_code, simplify_matrix

def jacobian(ndim,
             element_type="Lagrange",
             order=1,
             return_J=True,
             return_inv=True,
             return_det=True):
    """
    Symbolically compute the jacobian of the isoparametric mapping.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    J : symfem.functions.MatrixFunction
        jacobian of isoparametric mapping.
    Jinv : symfem.functions.MatrixFunction
        inverse of jacobian of isoparametric mapping.
    Jdet : symfem.functions.MatrixFunction
        determinant of jacobian of isoparametric mapping.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    #
    gradN = VectorFunction(basis).grad(ndim)
    #
    scaled = scale_cell(vertices)
    print(scaled)
    print(basis)
    print(gradN)
    print(gradN.transpose())
    #
    J = simplify_matrix( gradN.transpose()@scaled )
    if return_det or return_inv:
        Jdet = J.det()
    if return_inv:
        # adjungate matrix
        Jinv = [[[] for j in range(ndim)] for j in range(ndim)]
        if ndim == 2:
            Jinv[0][0], Jinv[1][1] = J[1][1]/Jdet, J[0][0]/Jdet
            Jinv[0][1], Jinv[1][0] = -J[0][1]/Jdet, -J[1][0]/Jdet
        elif ndim == 3:
            #
            Jinv[0][0] = (J[1][1]*J[2][2] - J[1][2]*J[2][1]) / Jdet
            Jinv[0][1] = -(J[0][1]*J[2][2] - J[0][2]*J[2][1]) / Jdet
            Jinv[0][2] = (J[0][1]*J[1][2] - J[0][2]*J[1][1]) / Jdet
            #
            Jinv[1][0] = -(J[1][0]*J[2][2] - J[1][2]*J[2][0]) / Jdet
            Jinv[1][1] = (J[0][0]*J[2][2] - J[0][2]*J[2][0] ) / Jdet
            Jinv[1][2] = -(J[0][0]*J[1][2] - J[0][2]*J[1][0]) / Jdet
            #
            Jinv[2][0] = (J[1][0]*J[2][1] - J[1][1]*J[2][0]) / Jdet
            Jinv[2][1] = -(J[0][0]*J[2][1] - J[0][1]*J[2][0]) / Jdet
            Jinv[2][2] = (J[0][0]*J[1][1] - J[0][1]*J[1][0]) / Jdet
        #
        Jinv = simplify_matrix( MatrixFunction(Jinv) )
    if all([return_J, not return_inv, not return_det]):
        return J
    elif all([not return_J, return_inv, not return_det]):
        return Jinv
    elif all([not return_J, not return_inv, return_det]):
        return Jdet
    elif all([return_J, return_inv, not return_det]):
        return J, Jinv
    elif all([return_J, not return_inv, return_det]):
        return J, Jdet
    elif all([not return_J, return_inv, return_det]):
        return Jinv, Jdet
    elif all([return_J,return_inv,return_det]):
        return J, Jinv, Jdet
    else:
        raise ValueError("At least on of the return options must be True.")

if __name__ == "__main__":


    #
    J, Jinv, Jdet = jacobian(ndim = 2)
    print(J,Jinv,Jdet)
    #print(convert_to_code(jacobian(ndim = 2)))
