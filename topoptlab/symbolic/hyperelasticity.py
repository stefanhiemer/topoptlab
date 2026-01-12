# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix

def stvenant_engdensity(E_v : Union[None,MatrixFunction] = None,
                        c : Union[None,MatrixFunction] = None,
                        ndim: Union[int,None] = 3) -> ScalarFunction:
    """
    Returns elastic energy density for St. Venant material defined as 
    
        engdensity = 1/2 E_v.T@C@E_v
    
    where E_v is the Cauchy strain in Voigt notation.

    Parameters
    ----------
    E_v : None or MatrixFunction
        Cauchy strain in Voigt notation. 
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
    if E_v is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E_v is None:
        E_v = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", symmetric=True) 
    return ScalarFunction( (E_v.transpose()@c@E_v / 2)[0,0].as_sympy() )

def stvenant_2PK(E_v : Union[None,MatrixFunction] =  None,
                 c : Union[None,MatrixFunction] = None,
                 ndim: Union[int,None] = 3) -> ScalarFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S for St. Venant material defined 
    as 
    
        S = C@E_v
    
    where E_v is the Cauchy strain in Voigt notation.

    Parameters
    ----------
    E_v : None or MatrixFunction
        Cauchy strain in Voigt notation. 
    c : None or MatrixFunction
        stiffness tensor in Voigt notation.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        symbolic 2. Piola-Kirchhoff stress (2PK).

    """
    if E_v is None and c is None and ndim is None:
        raise ValueError("All arguments are none. ",
                         "Either ndim must be an integer or the other two "
                         "arguments must be given.")
    #
    if E_v is None:
        E_v = generate_constMatrix(ncol=1,
                                   nrow=int((ndim**2 + ndim) /2),
                                   name="E")
    # anisotropic stiffness tensor or equivalent in Voigt notation 
    if c is None:
        c = generate_constMatrix(ncol=int((ndim**2 + ndim) /2),
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="c", 
                                 symmetric=True) 
    return c@E_v

def calculate_2PK(eng_density: ScalarFunction,
                  E_v : Union[None,MatrixFunction],
                  ndim: Union[int,None] = 3) -> MatrixFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) Sof a given elastic energy density 
    e by 
    
    
        S_i = d e / d E_i
    
    where E_i is the i-th entry of the Cauchy strain in Voigt notation.

    Parameters
    ----------
    E_v : None or MatrixFunction
        Cauchy strain in Voigt notation. 
    c : None or MatrixFunction
        stiffness tensor in Voigt notation.  
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    S : symfem.functions.MatrixFunction
        symbolic 2. Piola-Kirchhoff stress (2PK).

    """
    if E_v is None and ndim is None:
        raise ValueError("E_v and ndim are none. ",
                         "Either ndim must be an integer or E_v must be given.")
    #
    if E_v is None:
        E_v = generate_constMatrix(ncol=1,
                                   nrow=int((ndim**2 + ndim) /2),
                                   name="E")
    #
    S = []
    for i in range(E_v.shape[0]):
        S.append([eng_density.diff(variable=E_v[i,0])])
    return MatrixFunction(S)