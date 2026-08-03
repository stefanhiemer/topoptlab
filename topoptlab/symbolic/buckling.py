# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Union 

from symfem.symbols import x,t
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import eye, simplify_matrix,\
                                            generate_constMatrix,\
                                            generate_FunctMatrix, to_voigt,\
                                            from_voigt, to_column, kron,\
                                            integrate, to_square
from topoptlab.symbolic.strain_measures import dispgrad_matrix,\
                                               small_strain_matrix
from topoptlab.symbolic.stress_conversions import cauchy_to_pk1, pk2_to_pk1
#from topoptlab.symbolic.hyperelasticity import stvenant_2pk, stvenant_cauchy

def buckling_matrix(ndim : int,
                    u : Union[None,MatrixFunction],
                    c : Union[None,MatrixFunction],
                    element_type : str ="Lagrange",
                    order : int = 1) -> MatrixFunction:
    """
    Symbolically compute the tangent stiffness matrix for nonlinear elasticity.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    u : None or symfem.functions.MatrixFunction
        displacements at current iterate.
    c : None or symfem.functions.MatrixFunction
        stiffness tensor . if None, generic stiffness tensor is assumed.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    Ke : symfem.functions.MatrixFunction
        symbolic tangent stiffness matrix.
    fe : symfem.functions.MatrixFunction
        symbolic internal forces.

    """
    #
    vertices, nd_inds, ref, basis = base_cell(ndim)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c")
    #
    if u is None:
        u = generate_constMatrix(ncol=1,
                                 nrow=len(nd_inds)*ndim,
                                 name="u")
    # calculate matrix for deformation gradient
    b_h = dispgrad_matrix(ndim=ndim,
                          nd_inds=nd_inds,
                          basis=basis,
                          isoparam_kws={"element_type": element_type,
                                        "order": order})
    # calculate small strain matrix and stress
    b = small_strain_matrix(ndim=ndim,
                            nd_inds=nd_inds,
                            basis=basis,
                            isoparam_kws={"element_type": element_type,
                                          "order": order})
    sigma = kron(eye(ndim),from_voigt(b@u, eng_conv=False))
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    return simplify_matrix(M=integrate(M=b_h.transpose()@S@b_h,
                                       domain=ref,
                                       variables=x,
                                       dummy_vars=t, 
                                       parallel=None))

if  __name__ == "__main__":
    #res,symbs = residual(ndim=2,
    #                sigma=stvenant_cauchy,
    #                return_symbols=True)
    from topoptlab.symbolic.stvenant import stvenant_matmodel
    ndim = 2
    Ke,fe = tangentstiffness_matrix(ndim = ndim,
                                    u = None,
                                    material_model = stvenant_matmodel,
                                    material_constants = {"c": generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                                                                    nrow=int((ndim**2 + ndim) /2),
                                                                                    name="c")},
                                    plane_stress = False,
                                    element_type ="Lagrange",
                                    order = 1)
    print(fe)
    print()
    print(Ke)
    #print(linearize(res, symbols=symbs).shape)