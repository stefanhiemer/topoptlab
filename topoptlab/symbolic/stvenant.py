# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, is_voigt,\
                                            to_voigt
from topoptlab.symbolic.strain_measures import lagrangian_strain
from topoptlab.symbolic.stress_conversions import pk2_to_pk1, pk2_to_cauchy

def stvenant_engdensity(E : Union[None,MatrixFunction] = None,
                        F : Union[None,MatrixFunction] = None,
                        c : Union[None,MatrixFunction] = None,
                        ndim: Union[int,None] = 3, 
                        **kwargs: Any) -> ScalarFunction:
    """
    Returns elastic energy density for St. Venant material defined as 
    
        engdensity = 1/2 E.T@C@E
    
    where E is the Lagrangian strain in Voigt notation.

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
    engdensity : symfem.functions.ScalarFunction
        elastic energy density for St. Venant material.

    """
    #
    if E is None and F is not None:
        E = lagrangian_strain(ndim=ndim,
                              F=F)
    #
    if E is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    if not is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", symmetric=True) 
    return ScalarFunction( (E.transpose()@c@E / 2)[0,0].as_sympy() )

def stvenant_2pk(E : Union[None,MatrixFunction] =  None,
                 F : Union[None,MatrixFunction] = None,
                 c : Union[None,MatrixFunction] = None,
                 ndim: Union[int,None] = 3, 
                 **kwargs: Any) -> ScalarFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S for St. Venant material defined 
    as 
    
        S = C@E
    
    where E is the Lagrangian strain in Voigt notation.

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
    if E is None and F is not None:
        E = lagrangian_strain(ndim=ndim,
                              F=F)
    #
    if E is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    if not is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", 
                                 symmetric=True)
    return c@E

def stvenant_1pk(F : Union[None,MatrixFunction],
                 E : Union[None,MatrixFunction] =  None,
                 c : Union[None,MatrixFunction] = None,
                 ndim: Union[int,None] = 3, 
                 **kwargs: Any) -> ScalarFunction:
    """
    Returns 1. Piola-Kirchhoff stress (1PK) P for St. Venant material defined 
    by converting the expression for the 2PK. This function needs the 
    deformation gradient for the conversion.

    Parameters
    ----------
    F : None or symfem.functions.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim). If E is None, then 
        then E is calculated with the provided F.
    E : None or MatrixFunction
        Lagrangian strain in Voigt notation. 
    c : None or MatrixFunction
        stiffness tensor in Voigt notation.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        symbolic 1. Piola-Kirchhoff stress (1PK) in flattened notation.

    """
    #
    if E is None and F is not None:
        E = lagrangian_strain(ndim=ndim,
                              F=F)
    #
    if E is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    if not is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", 
                                 symmetric=True) 
    return pk2_to_pk1(S=stvenant_2pk(E=E,
                                     F=F,
                                     c=c,
                                     ndim=ndim), 
                      F=F) 

def stvenant_cauchy(F : Union[None,MatrixFunction],
                    Fdet: Union[None,ScalarFunction] = None,
                    E : Union[None,MatrixFunction] =  None,
                    c : Union[None,MatrixFunction] = None,
                    ndim: Union[int,None] = 3, 
                    **kwargs: Any) -> ScalarFunction:
    """
    Returns Cauchy stress for St. Venant material defined 
    by converting the expression for the 2PK. This function needs the 
    deformation gradient for the conversion.

    Parameters
    ----------
    F : None or symfem.functions.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim). If E is None, then 
        then E is calculated with the provided F.
    Fdet : None or symfem.functions.ScalarFunction
        determinant of deformation gradient.
    E : None or MatrixFunction
        Lagrangian strain in Voigt notation. 
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
    if E is None and F is not None:
        E = lagrangian_strain(ndim=ndim,
                              F=F)
    #
    if E is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    if not is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", 
                                 symmetric=True) 
    return pk2_to_cauchy(S=stvenant_2pk(E=E,
                                        F=F,
                                        c=c,
                                        ndim=ndim), 
                          F=F)