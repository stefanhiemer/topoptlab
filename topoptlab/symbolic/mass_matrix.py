# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.functions import MatrixFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.parametric_map import jacobian 
from topoptlab.symbolic.matrix_utils import simplify_matrix

def mass(scalarfield: bool,
         ndim: int,
         element_type: str = "Lagrange",
         order: int = 1,
         return_integrand: bool = False) -> MatrixFunction:
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
    return_integrand : bool
        if True, return the integrand (including the determinant of the
        jacobian) without integrating it over the element.

    Returns
    -------
    mass_matrix : symfem.functions.MatrixFunction
        symbolic mass matrix.

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
    if return_integrand:
        return integrand
    return simplify_matrix( integrand.integral(ref,x)) 
