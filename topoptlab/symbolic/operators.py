# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from symfem.functions import VectorFunction,MatrixFunction
from symfem.symbols import x

from topoptlab.symbolic.cell import base_cell 
from topoptlab.symbolic.matrix_utils import  generate_constMatrix,simplify_matrix 
from topoptlab.symbolic.parametric_map import jacobian

def aniso_laplacian(ndim: int, K: Union[None,MatrixFunction] = None,
                    element_type: str = "Lagrange",
                    order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the stiffness matrix for an anisotropic Laplacian 
    operator nabla K nabla. This type of operator is encountered in heat 
    conduction, diffusion, etc.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    K : None or symfem.functions.MatrixFunction
        heat conductivity tensor.
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
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    # anisotropic heat conductivity or equivalent
    if K is None:
        K = generate_constMatrix(ndim,ndim,"k")
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    integrand = gradN@K@gradN.transpose() * Jdet
    return simplify_matrix( integrand.integral(ref,x)) 