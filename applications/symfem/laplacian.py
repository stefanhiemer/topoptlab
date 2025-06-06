from symfem.functions import ScalarFunction,VectorFunction
from symfem.symbols import x
from sympy import symbols

from topoptlab.symfem_utils import base_cell ,generate_constMatrix
from topoptlab.symfem_utils import convert_to_code, jacobian, simplify_matrix

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
