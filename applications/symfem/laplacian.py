from symfem.functions import VectorFunction
from symfem.symbols import x

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
    print("1D iso\n",
          convert_to_code(iso_laplacian(ndim = 1)),"\n")
    print("2D iso\n",
          convert_to_code(iso_laplacian(ndim = 2)),"\n")
    print("3D iso\n",
          convert_to_code(iso_laplacian(ndim = 3)),"\n")
    print("1D aniso\n",
          convert_to_code(aniso_laplacian(ndim = 1),matrices = ["k"]),"\n")
    print("2D aniso\n",
          convert_to_code(aniso_laplacian(ndim = 2),matrices = ["k"]),"\n")
    print("3D aniso\n",
          convert_to_code(aniso_laplacian(ndim = 3),matrices = ["k"]),"\n")
