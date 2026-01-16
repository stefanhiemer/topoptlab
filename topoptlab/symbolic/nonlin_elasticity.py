# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Union 

from symfem.symbols import x
from symfem.functions import MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian
from topoptlab.symbolic.matrix_utils import eye, simplify_matrix,\
                                            inverse, generate_constMatrix,\
                                            generate_FunctMatrix, to_voigt,\
                                            from_voigt, to_column
from topoptlab.symbolic.strain_measures import cauchy_strain,dispgrad_matrix, def_grad
from topoptlab.symbolic.stress_conversions import cauchy_to_pk1, pk2_to_pk1
from topoptlab.symbolic.hyperelasticity import stvenant_2pk, stvenant_cauchy

def residual(ndim : int,
             sigma : Union[None,Callable],
             pk1: Union[None,Callable] = None,
             pk2: Union[None,Callable] = None,
             material_params: Dict = dict(),
             element_type : str ="Lagrange",
             order : int = 1,
             return_symbols: bool =False) -> MatrixFunction:
    """
    Symbolically compute the residual for nonlinear elasticity formulated in 
    terms of the 1. Piola-Kirchhoff (PK) stress P and the deformation gradient
    F.
    
    The constitutive behaviouris given by a stress relationship supplied via a 
    Callable of the stress. You must decide in which stress the material law is 
    formulated and depending on that the solver converts the given callable to 
    calculate the stress to the 1. PK. The callable should calculate the stress
    based on the deformation gradient F at the current iteration with nodal 
    displacements u0.

    Parameters
    ----------
    ndim : int
        number of spatial dimensions. Must be between 1 and 3.
    sigma : None or callable
        returns Cauchy stress in Voigt form. This is what an Elmer UMAT should 
        do.
    pk1 : None or callable
        returns 1. PK stress in flattened shape. 
    pk2 : None or callable
        returns 2. PK stress in Voigt form. 
    material_params : Dict,
        dictionary containing material parameters.
    element_type : str
        type of element.
    order : int
        order of element.

    Returns
    -------
    stiffness_matrix : symfem.functions.MatrixFunction
        symbolic residual of shape (n_nodes,1).

    """
    #
    vertices, nd_inds, ref, basis  = base_cell(ndim)
    #
    u,u_symbs = generate_constMatrix(ncol=1, nrow=len(basis)*ndim, name="u",
                                     return_symbols=True)
    # calculate deformation gradient
    b_h = dispgrad_matrix(ndim=ndim,
                          nd_inds=nd_inds,
                          basis=basis,
                          isoparam_kws={"element_type": element_type,
                                        "order": order}) 
    F = def_grad(ndim=ndim,
                 u = u,
                 element_type = "Lagrange",
                 order = order,
                 shape = "square")
    # 
    if sigma is not None:
        P = cauchy_to_pk1(sigma=from_voigt(sigma(u_symbs=u_symbs,
                                                 ndim=ndim,
                                                 F=F,
                                                 **material_params)),
                          F=F,
                          Fdet=None)
    elif pk1:
        P = pk1(F=F,**material_params)
    elif pk2:
        P = pk2_to_pk1(S=pk2(F=F,**material_params), 
                       F=F)
    else:
        raise NotImplementedError()
    # create full integral and multiply with determinant
    Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                    return_J=False, return_inv=False, return_det=True)
    integrand = b_h.transpose()@ to_column(P) * Jdet
    if return_symbols:
        return simplify_matrix( integrand.integral(ref,x) ), u_symbs
    else:
        return simplify_matrix( integrand.integral(ref,x) )

def linearize(r: MatrixFunction,
              symbols: List) -> MatrixFunction:
    """
    For the given symbolic residual, take first derivative with respect to 
    the given symbols for linearization yielding the jacobian of the residual.

    Parameters
    ----------
    r : int
        number of spatial dimensions. Must be between 1 and 3.
    symbols : list
        list of sympy symbols.

    Returns
    -------
    J : symfem.functions.MatrixFunction
        jacobian of symbolic residual of shape (n_nodes,n_symbols).

    """
    #
    n = r.shape[0]
    #
    K = []
    for symb in symbols:
        K.append(r.diff(symb))
    return simplify_matrix(MatrixFunction([[K[j][i,0] for j in range(n)]\
                                            for i in range(n)])) 

def trial_cauchy(u_symbs,ndim,**kwargs):
    #
    sigma = generate_FunctMatrix(ncol=ndim,
                                 nrow=ndim,
                                 variables=u_symbs,
                                 name="s",symmetric=True)
    #
    return to_voigt(sigma)

if  __name__ == "__main__":
    res,symbs = residual(ndim=2,
                    sigma=stvenant_cauchy,
                    return_symbols=True)
    print(linearize(res, symbols=symbs).shape)