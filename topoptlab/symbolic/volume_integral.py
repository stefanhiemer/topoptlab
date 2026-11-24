# SPDX-License-Identifier: GPL-3.0-or-later
from symfem.functions import MatrixFunction
from symfem.symbols import x, t

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.matrix_utils import integrate
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.parametric_map import jacobian

def volume_integral(ndim: int,
                    element_type: str = "Lagrange",
                    order: int = 1,
                    return_integrand: bool = False,
                    backend: str = "sympy") -> MatrixFunction:
    """
    Symbolically compute the integral of the shape functions over an
    element, i.e. the row vector w such that for nodal values u the volume
    integral of the interpolated field is given by w @ u:

        integral_Omega N dOmega.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    element_type : str
        type of element.
    order : int
        order of element.
    return_integrand : bool
        if True, return the integrand (including the determinant of the
        jacobian) without integrating it over the element.
    backend : str
        integration backend, either "sympy" or "symint". See
        topoptlab.symbolic.scalar_utils.integrate_expression.

    Returns
    -------
    volume_weights : symfem.functions.MatrixFunction
        symbolic row vector of shape (1,n_nodes) containing the integral of
        each shape function over the element.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim=ndim,
                                               element_type=element_type,
                                               order=order)
    #
    N = shape_function_matrix(basis=basis,nedof=1,mode="col")
    # get shape functions as a column vector/matrix and multiply with
    # determinant of jacobian of isoparametric mapping
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = N.transpose()*Jdet
    if return_integrand:
        return integrand
    return integrate(M=integrand, domain=ref, variables=x, dummy_vars=t,
                     backend=backend)
