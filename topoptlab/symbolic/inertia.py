# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union
from math import floor

from sympy import symbols
from symfem.functions import VectorFunction,MatrixFunction
from symfem.symbols import x,t

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.strain_measures import jacobian_matrix
from topoptlab.symbolic.matrix_utils import generate_constMatrix,\
                                            generate_FunctMatrix,\
                                            simplify_matrix,kron,eye,\
                                            from_vectorfunction,flatten,\
                                            inverse, integrate, ones, to_square
from topoptlab.symbolic.parametric_map import jacobian

def inertia(ndim: int, 
            v: Union[None,MatrixFunction] = None,
            element_type: str = "Lagrange",
            order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the convective term and linearized inertia matrix of the
    navier stokes equations

         v dot grad v

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    v : None or symfem.functions.MatrixFunction
        velocity.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    inertia_matrix : symfem.functions.MatrixFunction
        symbolic inertia matrix.
    convective_term : symfem.functions.MatrixFunction
        symbolic convective term (v·∇v discretised).

    """

    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    #
    if v is None:
        v = generate_constMatrix(ncol=1, nrow=ndim*len(basis), name="v")
    I = ones(nrow=v.shape[0],ncol=1)
    # shape functions for velocities
    N_v = shape_function_matrix(basis=basis,
                                nedof=ndim,
                                mode="col")
    #
    B_j = jacobian_matrix(ndim=ndim,
                          nd_inds=nd_inds,
                          basis=basis,
                          isoparam_kws={"element_type": element_type,
                                        "order": order})
    #
    Jdet = jacobian(ndim=ndim, 
                    element_type=element_type, 
                    order=order,
                    return_J=False, 
                    return_inv=False, 
                    return_det=True)
    #
    oseen = N_v@to_square(v=B_j@v, order="C")@N_v.transpose() 
    #
    v_h = N_v.transpose()@v
    newt_corr = N_v@kron(eye(ndim), v_h.transpose())@B_j
    #
    inertia_matrix = oseen + newt_corr
    #
    inertia_matrix = simplify_matrix(integrate(M=inertia_matrix*Jdet,
                                               domain=ref,
                                               variables=x,
                                               dummy_vars=t,
                                               parallel=None,
                                               symmetry=False))
    convective_term = N_v@to_square(v=B_j@v, order="C")@N_v.transpose()@v
    convective_term = simplify_matrix(integrate(M=convective_term*Jdet,
                                                domain=ref,
                                                variables=x,
                                                dummy_vars=t,
                                                parallel=None,
                                                symmetry=False))
    return inertia_matrix, convective_term

if  __name__ == "__main__":
    ndim = 2
    Ke = inertia(ndim=ndim,
                 element_type="Lagrange",
                 order=1)
    print(Ke)