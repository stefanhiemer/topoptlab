# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from sympy import expand,simplify,Expr
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.cell import base_cell
from topoptlab.symbolic.matrix_utils import simplify_matrix, generate_constMatrix
from topoptlab.symbolic.matrix_utils import inverse

def cauchy_to_pk1(sigma: MatrixFunction, 
                  F: MatrixFunction,
                  Fdet: Union[None,ScalarFunction] = None) -> MatrixFunction:
    """
    Convert Cauchy stress `sigma` to first Piola-Kirchhoff stress `P`:
        
        `P = det(F) sigma@F^{-1}.T`

    Parameters
    ----------
    sigma : symfem.functions.MatrixFunction
        Cauchy stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        first Piola-Kirchhoff stress tensor of shape (ndim,ndim).

    """
    #
    if Fdet is None:
        Fdet = F.det() 
    #
    return simplify_matrix( (sigma@inverse(F).transpose()).__mul__(Fdet) )

def cauchy_to_pk2(sigma: MatrixFunction, 
                  F: MatrixFunction,
                  Fdet: Union[None,ScalarFunction] = None,
                  Finv: Union[None,MatrixFunction] = None) -> MatrixFunction:
    """
    Convert Cauchy stress `sigma` to second Piola-Kirchhoff stress `S`:
        
        `S = det(F) F^{-1}@sigma@F^{-1}.T`

    Parameters
    ----------
    sigma : symfem.functions.MatrixFunction
        Cauchy stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.
    Finv : None or symfem.functions.MatrixFunction
        inverse of deformation gradient.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        second Piola-Kirchhoff stress tensor of shape (ndim,ndim).

    """
    #
    if Fdet is None:
        Fdet = F.det()
    #
    if Finv is None:
        Finv = inverse(F)
    #
    return simplify_matrix((Finv@sigma@Finv.transpose()).__mul__(Fdet))

def pk1_to_cauchy(P: MatrixFunction, 
                  F: MatrixFunction,
                  Fdet: Union[None,ScalarFunction] = None) -> MatrixFunction:
    """
    Convert first Piola-Kirchhoff stress `P` to Cauchy stress `sigma`:
        
        `sigma = P@F.T / det(F)`

    Parameters
    ----------
    P : symfem.functions.MatrixFunction
        first Piola-Kirchhoff stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.

    Returns
    -------
    sigma : symfem.functions.MatrixFunction
        Cauchy stress tensor of shape (ndim,ndim).

    """
    #
    if Fdet is None:
        Fdet = F.det() 
    #
    return simplify_matrix( (P@F.transpose()).__mul__(1/Fdet) )

def pk1_to_pk2(P: MatrixFunction, 
               F: MatrixFunction,
               Finv: Union[None,MatrixFunction] = None) -> MatrixFunction:
    """
    Convert first Piola-Kirchhoff stress `P` to second Piola-Kirchhoff 
    stress `S`:
        
        `S = F^{-1}@P`

    Parameters
    ----------
    P : symfem.functions.MatrixFunction
        first Piola-Kirchhoff stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Finv : None or symfem.functions.MatrixFunction
        inverse of deformation gradient.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        second Piola-Kirchhoff stress tensor of shape (ndim,ndim).

    """
    #
    if Finv is None:
        Finv = inverse(F) 
    #
    return simplify_matrix( Finv@P )

def pk2_to_cauchy(S: MatrixFunction, 
                  F: MatrixFunction,
                  Fdet: Union[None,ScalarFunction] = None) -> MatrixFunction:
    """
    Convert second Piola-Kirchhoff stress `S` to Cauchy stress `sigma`:
        
        `sigma = F@S@F.T / det(F)`

    Parameters
    ----------
    S : symfem.functions.MatrixFunction
        second Piola-Kirchhoff stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.

    Returns
    -------
    sigma : symfem.functions.MatrixFunction
        Cauchy stress tensor of shape (ndim,ndim).

    """
    #
    if Fdet is None:
        Fdet = F.det() 
    #
    return simplify_matrix( (F@S@F.transpose()).__mul__(1/Fdet) )

def pk2_to_pk1(S: MatrixFunction, 
               F: MatrixFunction) -> MatrixFunction:
    """
    Convert second Piola-Kirchhoff stress `S` to first Piola-Kirchhoff 
    stress `P`:
        
        `P = F@S`

    Parameters
    ----------
    S : symfem.functions.MatrixFunction
        second Piola-Kirchhoff stress tensor of shape (ndim,ndim).
    F : symfem.functions.MatrixFunction
        deformation gradient of shape (ndim,ndim).
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        first Piola-Kirchhoff stress tensor of shape (ndim,ndim).

    """
    #
    return simplify_matrix( F@S )
