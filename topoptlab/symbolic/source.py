# SPDX-License-Identifier: GPL-3.0-or-later

from symfem.symbols import x,t
from symfem.functions import ScalarFunction, MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import simplify_matrix, \
                                            generate_constMatrix, \
                                            integrate

def source(ndim : int,
           volume_source : None | ScalarFunction = None,
           element_type : str = "Lagrange",
           order : int = 1) -> MatrixFunction:
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
    source vector : MatrixFunction
        symbolic stiffness matrix as list of lists .

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    #
    if volume_source is None:
        volume_source = generate_constMatrix(ncol=1,nrow=1,name="b")
    #
    N = shape_function_matrix(basis=basis, nedof=1, mode="col")
    # get shape functions as a column vector/matrix and multiply with
    # determinant of jacobian of isoparametric mapping
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N@volume_source*Jdet
    return simplify_matrix(integrate(M=integrand,
                                     domain=ref,
                                     variables=x,
                                     dummy_vars=t, 
                                     parallel=None, 
                                     symmetry=False), 
                           symmetry=True)