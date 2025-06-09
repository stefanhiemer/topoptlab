from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symfem_utils import base_cell, shape_function_matrix
from topoptlab.symfem_utils import convert_to_code, jacobian

def mass(scalarfield,
         ndim,
         element_type="Lagrange",
         order=1):
    """
    Symbolically compute the mass matrix.

    Parameters
    ----------
    scalarfield : bool
        if True, assume scalarfield. If not assume vector field.
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
    if scalarfield:
        N = MatrixFunction([[b] for b in basis])
    else:
        N = shape_function_matrix(basis=basis,nedof=ndim,mode="col")
    # create integral
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N@N.transpose() * Jdet
    return integrand.integral(ref,x)

if __name__ == "__main__":


    #
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(mass(scalarfield=True,
                                   ndim = dim),vectors=["l"]),"\n")
