# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.functions import VectorFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell 
from topoptlab.symbolic.matrix_utils import  generate_constMatrix,simplify_matrix
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.parametric_map import jacobian

def iso_laplacian(ndim,
                  element_type="Lagrange",
                  order=1):
    """
    Symbolically compute the stiffness matrix for the Laplacian operator.

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
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix as list of lists .

    """

    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    integrand = gradN@gradN.transpose() * Jdet
    return simplify_matrix( integrand.integral(ref,x))

def aniso_laplacian(ndim,
                    element_type="Lagrange",
                    order=1):
    """
    Symbolically compute the stiffness matrix for the Laplacian operator.

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
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix as list of lists .

    """

    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    # anisotropic heat conductivity or equivalent
    K = generate_constMatrix(ndim,ndim,"k")
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    integrand = gradN@K@gradN.transpose() * Jdet
    return simplify_matrix( integrand.integral(ref,x))

if __name__ == "__main__":

    #
    #for dim in range(1,4):
    #    print(str(dim)+"D")
    #    print(convert_to_code(iso_laplacian(ndim = dim),
    #                          matrices=["k"],vectors=["l","g"]),"\n")
    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(aniso_laplacian(ndim = dim),
                              matrices=["k"],vectors=["l","g"]),"\n")
