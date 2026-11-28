# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

from sympy import ln, Rational
from sympy.core.symbol import Symbol,Expr
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, \
                                            is_voigt,\
                                            to_voigt, to_square, trace, \
                                            inverse, eye
from topoptlab.symbolic.strain_measures import cauchy_strain

def neohookean_engdensity(F : None | MatrixFunction = None,
                          h : None | Symbol | ScalarFunction = None,
                          mu: None | Symbol | ScalarFunction = None,
                          Fdet: None | Expr | ScalarFunction = None,
                          ndim: int | None = 3, 
                          **kwargs: Any) -> ScalarFunction:
    """
    Return elastic energy density for Neo-Hookean material defined as 
    
        engdensity = h/2*ln(det(F))**2 + mu*((tr(C) - 3)/2 - ln(det(F))) 
    
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
        elastic energy density for Neo-Hookean material.

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
    return ScalarFunction( h/2*ln( Fdet )**2 + mu*( Rational(1,2)*( trace(C,mode="sympy")-3) - ln(Fdet))  )

def neohookean_2pk(F : None | MatrixFunction = None,
                   h : None | Symbol | ScalarFunction = None,
                   mu: None | Symbol | ScalarFunction = None,
                   Fdet: None | ScalarFunction = None,
                   ndim: int | None = 3, 
                   **kwargs: Any) -> MatrixFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S for Neo-Hookean material:
        
        S = h*ln(det(F))*inv(C) + mu*(I-inv(C))
        
    
    where h and mu are the Lame constants, det(F) the determinant of the 
    deformation gradient F, C = F.T@F the right Cauchy–Green deformation 
    tensor, I the identity and inv the matrix inverse.

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
    Cinv = inverse(cauchy_strain(ndim=F.shape[0],F=F))
    # first Lame constant 
    if h is None:
        h =  Symbol("h")
    # second Lame constant
    if mu is None:
        mu =  Symbol("mu")
    return to_voigt(Cinv.__mul__(h*ln(Fdet)) + \
                    (eye(Cinv.shape[0])-Cinv).__mul__(mu), 
                    eng_conv=False)

def neohookean_1pk(F : None | MatrixFunction = None,
                   h : None | Symbol | ScalarFunction = None,
                   mu: None | Symbol | ScalarFunction = None,
                   Fdet: None | ScalarFunction = None,
                   ndim: int | None = 3, 
                   **kwargs: Any) -> MatrixFunction:
    """
    Returns 1. Piola-Kirchhoff stress (1PK) P for Neo-Hookean material:
        
        P = h*ln(det(F))*inv(F).T + mu*(F-inv(F).T)
        
    
    where h and mu are the Lame constants, F the deformation gradient, inv the
    matrix inverse and det(F) the determinant of F.

    Parameters
    ----------
    F : None or MatrixFunction
        Deformation gradient either of shape (ndim,ndim) or flattened in 
        C-order of shape (ndim**2,1). 
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
        symbolic 1. Piola-Kirchhoff stress (1PK) of shape (ndim,ndim).

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
    # flattened (C-order) deformation gradient to matrix form
    elif F.shape[1] == 1 and F.shape[0] > 1:
        F = to_square(v=F, order="C")
    # 
    if Fdet is None:
        Fdet = F.det().as_sympy()
    elif isinstance(Fdet,ScalarFunction):
        Fdet = Fdet.as_sympy()
    # first Lame constant 
    if h is None:
        h =  Symbol("h")
    # second Lame constant
    if mu is None:
        mu =  Symbol("mu")
    FinvT = inverse(F.transpose())
    return FinvT.__mul__(h*ln(Fdet)) + (F-FinvT).__mul__(mu)
        

if __name__ == "__main__":
    print(neohookean_1pk(ndim=2))