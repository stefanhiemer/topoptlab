# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.symbols import x, t
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.matrix_utils import generate_constMatrix, simplify_matrix, integrate
from topoptlab.symbolic.parametric_map import jacobian

def body_force(ndim: int,
               element_type: str = "Lagrange",
               order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the body force.

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
    body_force = generate_constMatrix(ncol=1,nrow=ndim,name="b")
    N = shape_function_matrix(basis=basis,nedof=ndim,mode="col")
    # get shape functions as a column vector/matrix and multiply with
    # determinant of jacobian of isoparametric mapping
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N@body_force*Jdet
    return simplify_matrix(integrate(M=integrand,
                                     domain=ref,
                                     variables=x,
                                     dummy_vars=t, 
                                     parallel=None, 
                                     symmetry=True))
