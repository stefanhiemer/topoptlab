# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Union

from sympy import ln,exp,simplify
from sympy.core.symbol import Symbol,Expr
from symfem.symbols import x,t
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, \
                                            generate_FunctMatrix, \
                                            is_voigt,\
                                            to_voigt, trace, inverse, eye, \
                                            simplify_matrix,integrate
from topoptlab.symbolic.strain_measures import cauchy_strain,green_strain,def_grad
from topoptlab.symbolic.operators import hessian_matrix
from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.parametric_map import jacobian

def huhu_engdensity(u : Union[None,MatrixFunction], 
                    ndim: int,
                    F : Union[None,MatrixFunction] = None,
                    a : Union[None,Symbol,ScalarFunction] = None,
                    kr : Union[None,Symbol,ScalarFunction] = None,
                    Fdet: Union[None,Expr,ScalarFunction] = None,
                    element_type: str = "Lagrange",
                    order: int = 1,
                    do_integral: bool = True,
                    **kwargs: Any) -> ScalarFunction:
    """
    Return elastic energy density for HuHu regularization defined as 
    
        engdensity = kr/2*exp(-a * det(F)) (H@u)^T H@u 
    
    where det(F) is the determinant of the deformation gradient, u the nodal 
    displacements, H the hessian of the shape functions, kr the regularization
    strength and a an optional exponent. If a is None, then the expression 
    reduces to 
    
        engdensity = kr/2*(H@u)^T H@u  
    
    """
    #
    if element_type:
        vertices, nd_inds, ref, basis  = base_cell(ndim,
                                                   element_type=element_type,
                                                   order=order)
        Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                              return_J=False, return_inv=True, return_det=True)
    # regularization strength
    if kr is None:
        kr =  Symbol("kr")
    #
    B_hessian = hessian_matrix(scalarfield=False,ndim=ndim, 
                               integrate=False,
                               element_type=element_type, 
                               order=order)
    #
    if u is None:
        u = generate_constMatrix(ncol=1, nrow=B_hessian.shape[1], name="u")
    h = simplify_matrix(B_hessian@u)
    huhu = (h.transpose()@h)[0,0].as_sympy()/2
    # exponential
    if a is not None:
        #
        if F is None and element_type is None:
            F = generate_constMatrix(ncol=ndim,
                                     nrow=ndim,
                                     name="F")
        elif F is None and element_type is not None:
            F = def_grad(ndim=ndim,
                         u=u,
                         element_type=element_type,
                         order=order,
                         shape="square")
        # 
        if Fdet is None:
            Fdet = F.det().as_sympy()
        elif isinstance(Fdet,ScalarFunction):
            Fdet = Fdet.as_sympy()
        #
        huhu = exp(-a*simplify(Fdet) )*huhu 
    #
    huhu = ScalarFunction( huhu )
    #
    if element_type is not None and do_integral:
        huhu = huhu.integral(ref,x) 
    return simplify(huhu)

def huhu_tangent(u : Union[None,MatrixFunction], 
                 ndim: int,
                 F : Union[None,MatrixFunction] = None,
                 a : Union[None,Symbol,ScalarFunction] = None,
                 kr : Union[None,Symbol,ScalarFunction] = None,
                 Fdet: Union[None,Expr,ScalarFunction] = None,
                 element_type: str = "Lagrange",
                 order: int = 1,
                 do_integral: bool = True,
                 **kwargs: Any) -> MatrixFunction:
    """
    Return consistent tangent for HuHu regularization defined as 
    
        K_T = kr*exp(-a * det(F)) (H)^T H
    
    where det(F) is the determinant of the deformation gradient, u the nodal 
    displacements, H the hessian of the shape functions, kr the regularization
    strength and a an optional exponent. If a is None, then the expression 
    reduces to 
    
        engdensity = kr/2*(H@u)^T H@u  
    
    """
    #
    if element_type:
        vertices, nd_inds, ref, basis  = base_cell(ndim,
                                                   element_type=element_type,
                                                   order=order)
        Jinv, Jdet = jacobian(ndim=ndim, element_type=element_type, order=order,
                              return_J=False, return_inv=True, return_det=True)
    # regularization strength
    if kr is None:
        kr =  Symbol("kr")
    #
    B_hessian = hessian_matrix(scalarfield=False,ndim=ndim, 
                               integrate=False,
                               element_type=element_type, 
                               order=order) 
    #
    huhu = simplify_matrix(M=B_hessian.transpose()@B_hessian)
    # exponential
    if a is not None:
        #
        if F is None and element_type is None:
            F = generate_constMatrix(ncol=ndim,
                                     nrow=ndim,
                                     name="F")
        elif F is None and element_type is not None:
            F = def_grad(ndim=ndim,
                         u=u,
                         element_type=element_type,
                         order=order,
                         shape="square")
        # 
        if Fdet is None:
            Fdet = F.det().as_sympy()
        elif isinstance(Fdet,ScalarFunction):
            Fdet = Fdet.as_sympy()
        #
        huhu = exp(-a*simplify(Fdet) )*huhu
        huhu = simplify_matrix(huhu)
    print(huhu)
    import sys 
    sys.exit()
    #
    if element_type is not None and do_integral:
        huhu = integrate(M=huhu,
                         domain=ref,
                         variables=x,
                         dummy_vars=t, 
                         parallel=None) 
    return simplify_matrix(huhu)

if __name__ == "__main__":
    
    #print(neohookean_1pk(ndim=3).shape)
    print(huhu_tangent(u=None,ndim=2,
          a=Symbol("a")))