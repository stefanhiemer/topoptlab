# SPDX-License-Identifier: GPL-3.0-or-later

from symfem.functions import VectorFunction,MatrixFunction
from symfem.symbols import x,t

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.shapefunction_matrix import shape_function_matrix
from topoptlab.symbolic.matrix_utils import generate_constMatrix,\
                                            simplify_matrix,integrate
from topoptlab.symbolic.parametric_map import jacobian

def advection(ndim: int, 
              v: None | MatrixFunction = None,
              element_type: str = "Lagrange",
              velocity: bool = False,
              order: int = 1) -> MatrixFunction:
    """
    Symbolically compute the advection matrix given velocity field v
    
     v.T @ nabla phi,
    
    of a scalar field phi. If velocity is True, we instead compute the weak form needed
    for solving for the velocity

       (nabla phi).T@phi

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
    advection_matrix : symfem.functions.MatrixFunction
        symbolic advection matrix as list of lists .

    """

    #
    vertices, nd_inds, ref, basis  = base_cell(ndim,
                                               element_type=element_type,
                                               order=order)
    #
    if v is None and not velocity:
        v = generate_constMatrix(ncol=1, nrow=ndim*len(basis), name="v")
    else:
        phi = generate_constMatrix(ncol=1, nrow=len(basis), name="phi")
    # shape functions for scalar field phi
    N = shape_function_matrix(basis=basis,nedof=1,mode="col")
    # shape functions for velocities
    N_v = shape_function_matrix(basis=basis,nedof=ndim,mode="col")
    #
    Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                          return_J=False, return_inv=True, return_det=True)
    gradN = VectorFunction(basis).grad(ndim)@Jinv.transpose()
    #
    if velocity:
        integrand = N@phi.transpose()@gradN@N_v.transpose()
    else:
        integrand = N@v.transpose()@N_v@gradN.transpose()
    return simplify_matrix(integrate(M=integrand*Jdet,
                                     domain=ref,
                                     variables=x,
                                     dummy_vars=t,
                                     parallel=None,
                                     symmetry=False))

if  __name__ == "__main__":
    ndim = 2
    Ke = advection(ndim=ndim,
                   element_type="Lagrange",
                   order=1)
    print(Ke)