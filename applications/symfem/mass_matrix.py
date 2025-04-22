from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell,scale_cell
from topoptlab.symfem_utils import convert_to_code, jacobian

def mass(ndim,
         element_type="Lagrange",
         order=1):
    """
    Symbolically compute the mass matrix.

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
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # get shape functions as a column vector/matrix
    N = MatrixFunction([[b] for b in basis])
    # create integral
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N@N.transpose() * Jdet
    return integrand.integral(ref,x)

if __name__ == "__main__":


    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(mass(ndim = dim),vectors=["l"]),"\n")
