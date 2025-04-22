from symfem.functions import MatrixFunction
from symfem.symbols import x
from sympy import symbols

from topoptlab.symfem_utils import base_cell,shape_function_matrix,generate_constMatrix
from topoptlab.symfem_utils import convert_to_code, jacobian

def body_force(ndim,field="vector",
               element_type="Lagrange",
               order=1):
    """
    Symbolically compute the mass matrix.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    field : str
        either "scalar" or "vector"
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : list
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    #
    if field == "scalar":
        body_force = generate_constMatrix(ncol=1,nrow=1,name="b")
        N = shape_function_matrix(basis=basis, nedof=1, mode="col")
    else:
        body_force = generate_constMatrix(ncol=1,nrow=ndim,name="b")
        N = shape_function_matrix(basis=basis,nedof=ndim,mode="col")
    # get shape functions as a column vector/matrix and multiply with
    # determinant of jacobian of isoparametric mapping
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N@body_force*Jdet
    return integrand.integral(ref,x)

if __name__ == "__main__":


    #
    print("1D\n",convert_to_code(body_force(ndim = 1),vectors=["b"]),"\n")
    print("2D\n",convert_to_code(body_force(ndim = 2),vectors=["b"]),"\n")
    print("3D\n",convert_to_code(body_force(ndim = 3),vectors=["b"]),"\n")
