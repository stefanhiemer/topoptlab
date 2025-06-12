from symfem.functions import MatrixFunction
from symfem.symbols import x
from sympy import symbols

from topoptlab.symfem_utils import base_cell,shape_function_matrix,generate_constMatrix
from topoptlab.symfem_utils import convert_to_code, jacobian

def volume_integral(ndim,
                    element_type="Lagrange",
                    order=1):
    """
    Symbolically compute the volume integral of a nodal quantity.

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
    nodal_forces : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    #
    N = shape_function_matrix(basis=basis,nedof=1,mode="col")
    # get shape functions as a column vector/matrix and multiply with
    # determinant of jacobian of isoparametric mapping
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N.transpose()*Jdet
    return integrand.integral(ref,x)

if __name__ == "__main__":


    #
    for dim in range(1,4):
        print(str(dim)+"D")
        print(convert_to_code(volume_integral(ndim = dim),vectors=["u","l"]))
