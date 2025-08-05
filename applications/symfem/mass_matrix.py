from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.cell import shape_function_matrix
from topoptlab.symbolic.code_conversion import convert_to_code
from topoptlab.symbolic.parametric_map import jacobian 
from topoptlab.symbolic.matrix_utils import simplify_matrix

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
    return simplify_matrix( integrand.integral(ref,x))

if __name__ == "__main__":


    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(mass(scalarfield=False,
                                   ndim = dim),vectors=["l"]),"\n")
