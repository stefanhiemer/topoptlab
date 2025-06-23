from symfem.functions import MatrixFunction, ScalarFunction
from symfem.symbols import x
from sympy import Symbol, Q, Ne, Piecewise

from topoptlab.symfem_utils import base_cell, shape_function_matrix
from topoptlab.symfem_utils import convert_to_code, jacobian, simplify_matrix
from topoptlab.symfem_utils import generate_constMatrix

def monomial(mononomial_order,
             scalarfield,
             ndim,
             element_type="Lagrange",
             order=1):
    """
    Symbolically compute the mass matrix.

    Parameters
    ----------
    mononomial_order : int 
        order of the monomial
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
    if mononomial_order < 2:
        raise ValueError("polynomial_order below order 2 is not sensible in this function.")
    if not scalarfield:
        raise NotImplementedError("Polynomial for vectorfields not yet implemented.")
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # get shape functions as a column vector/matrix
    if scalarfield:
        N = MatrixFunction([[b] for b in basis])
    else:
        N = shape_function_matrix(basis=basis,nedof=ndim,mode="col")
    # get state variable u
    u = generate_constMatrix(nrow=N.shape[0],ncol=1,name="u",
                             return_symbols=False)
    # create integral
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    #
    integrand = N
    uhat = N.transpose()@u
    # safely
    for i in range(mononomial_order-1):
        integrand = integrand@uhat
    integrand = integrand@N.transpose()*Jdet
    return simplify_matrix( integrand.integral(ref,x) )

if __name__ == "__main__":
    #
    for dim in range(2,3):
        print(str(dim)+"D")
        print(convert_to_code(monomial(mononomial_order=2,
                                       scalarfield=True,
                                       ndim = dim),vectors=["l","u"]),"\n")
