# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union
from math import floor

from symfem.symbols import x
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import eye, simplify_matrix, generate_constMatrix
from topoptlab.symbolic.strain_measures import cauchy_strain,dispgrad_matrix 

def residual(ndim : int,
             s : Union[None,MatrixFunction],
             element_type : str ="Lagrange",
             order : int = 1) -> MatrixFunction:
    """
    Symbolically compute the residual for nonlinear 
    elasticity formulated in terms of the 1. PK stress P.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    s : None or symfem.functions.MatrixFunction
        stiffness tensor .
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic stiffness matrix.

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if s is None:
        s = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="s")
    #
    u0 = generate_constMatrix(ncol=1, nrow=len(basis), name="u0")
    # calculate deformation gradient
    I = eye(size=ndim)
    b_h = dispgrad_matrix(ndim=ndim,
                          nd_inds=nd_inds,
                          basis=basis,
                          isoparam_kws={"element_type": element_type,
                                        "order": order},
                          shape="voigt")
    defgrad = MatrixFunction([I[i%ndim,floor(i/ndim)] for i in range(ndim**2) ])\
              + b_h@u0 
    # calculate necessary parameters from def. grad.
    raise NotImplementedError("Got stopped before calculating inverse, det. etc. of def. grad.")
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b_h.transpose()@s@b_h * Jdet
    return simplify_matrix( integrand.integral(ref,x) )

