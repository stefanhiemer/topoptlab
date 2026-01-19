# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

from sympy import ln
from sympy.core.symbol import Symbol,Expr
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, is_voigt,\
                                            to_voigt, trace, inverse, eye
from topoptlab.symbolic.strain_measures import cauchy_strain,green_strain

def neohookean_engdensity(F : Union[None,MatrixFunction] = None,
                          h : Union[None,Symbol,ScalarFunction] = None,
                          mu: Union[None,Symbol,ScalarFunction] = None,
                          Fdet: Union[None,Expr,ScalarFunction] = None,
                          ndim: Union[int,None] = 3, 
                          **kwargs: Any) -> ScalarFunction:
    """
    Returns elastic energy density for Neo-Hookean material defined as 
    
        engdensity = h*ln(det(F))**2 + mu*((tr(C)/2 - 3) - ln(det(F))) 
    
    where F is the deformation gradient in matrix notation, C the
    Cauchy–Green deformation tensor, h and mu are the Lame constants, and 
    det(F) the determinant of F.

    Parameters
    ----------
    F : None or MatrixFunction
        Deformation gradient. 
    h : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        first Lame constant.
    mu : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        second constant.
    Fdet : None or sympy.Expr or ScalarFunction
        determinant of deformation gradient.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    engdensity : symfem.functions.ScalarFunction
        elastic energy density for St. Venant material.

    """ 
    #
    if F is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or F must be given.")
    #
    if F is None:
        F = generate_constMatrix(ncol=ndim,
                                 nrow=ndim,
                                 name="F")
    # 
    if Fdet is None:
        Fdet = F.det().as_sympy()
    elif isinstance(Fdet,ScalarFunction):
        Fdet = Fdet.as_sympy()
    #
    C = cauchy_strain(ndim=ndim,
                      F=F)
    # first Lame constant 
    if h is None:
        h =  Symbol("h")
    # second Lame constant
    if mu is None:
        mu =  Symbol("mu")
    return ScalarFunction( h*ln( Fdet )**2 + mu*( 1/2*( trace(C,mode="sympy")-3) - ln(Fdet))  )

def neohookean_2pk(F : Union[None,MatrixFunction] = None,
                   h : Union[None,Symbol,ScalarFunction] = None,
                   mu: Union[None,Symbol,ScalarFunction] = None,
                   Fdet: Union[None,ScalarFunction] = None,
                   ndim: Union[int,None] = 3, 
                   **kwargs: Any) -> MatrixFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S for Neo-Hookean material:
        
        S = h*ln(det(F))*inv(B) + mu*(1-inv(B))
        
    
    where h and mu are the Lame constants, det(F) the determinant of the 
    deformation gradient F, B the left Cauchy–Green deformation tensor and inv 
    the matrix inverse.

    Parameters
    ----------
    E : None or MatrixFunction
        Lagrangian strain in Voigt notation. 
    F : None or symfem.functions.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim). If E is None, then 
        then E is calculated with the provided F.
    c : None or MatrixFunction
        stiffness tensor in Voigt notation.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        symbolic 2. Piola-Kirchhoff stress (2PK) in Voigt notation.

    """
    #
    if F is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or F must be given.")
    #
    if F is None:
        F = generate_constMatrix(ncol=ndim,
                                 nrow=ndim,
                                 name="F")
    # 
    if Fdet is None:
        Fdet = F.det().as_sympy()
    elif isinstance(Fdet,ScalarFunction):
        Fdet = Fdet.as_sympy()
    #
    Binv = inverse(green_strain(ndim=ndim,F=F))
    # first Lame constant 
    if h is None:
        h =  Symbol("h")
    # second Lame constant
    if mu is None:
        mu =  Symbol("mu")
    return Binv.__mul__(h*ln(Fdet)) + (eye(Binv.shape[0])-Binv).__mul__(mu) 

def neohookean_1pk(F : Union[None,MatrixFunction] = None,
                   h : Union[None,Symbol,ScalarFunction] = None,
                   mu: Union[None,Symbol,ScalarFunction] = None,
                   Fdet: Union[None,ScalarFunction] = None,
                   ndim: Union[int,None] = 3, 
                   **kwargs: Any) -> MatrixFunction:
    """
    Returns 1. Piola-Kirchhoff stress (1PK) P for Neo-Hookean material:
        
        P = h*ln(det(F))*inv(F).T + mu*(F-inv(F).T)
        
    
    where h and mu are the Lame constants, F the deformation gradient, inv the
    matrix inverse and det(F) the determinant of F.

    Parameters
    ----------
    F : None or MatrixFunction
        Deformation gradient. 
    h : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        first Lame constant.
    mu : None or sympy.core.symbol.Symbol or symfem.functions.ScalarFunction
        second constant.
    Fdet : None or MatrixFunction
        determinant of deformation gradient.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        symbolic 1. Piola-Kirchhoff stress (1PK) in flattened notation.

    """
    #
    if F is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or F must be given.")
    #
    if F is None:
        F = generate_constMatrix(ncol=ndim,
                                 nrow=ndim,
                                 name="F")
    # 
    if Fdet is None:
        Fdet = F.det().as_sympy()
    elif isinstance(Fdet,ScalarFunction):
        Fdet = Fdet.as_sympy()
    #
    if is_voigt(M=F,ndim=ndim):
        F = to_voigt(F)
    # first Lame constant 
    if h is None:
        h =  Symbol("h")
    # second Lame constant
    if mu is None:
        mu =  Symbol("mu")
    FinvT = inverse(F.transpose())
    return FinvT.__mul__(h*ln(Fdet)) + (F-FinvT).__mul__(mu) 

def huhu_engdensity() -> ScalarFunction:
    
    return

if __name__ == "__main__":
    
    print(neohookean_1pk(ndim=3).shape)