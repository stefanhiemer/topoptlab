# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import generate_constMatrix, is_voigt,\
                                            to_voigt
from topoptlab.symbolic.strain_measures import lagrangian_strain

def calculate_1pk(eng_density: ScalarFunction,
                  F : Union[None,MatrixFunction] = None,
                  ndim: Union[int,None] = 3, 
                  **kwargs: Any) -> MatrixFunction:
    """
    Returns 1. Piola-Kirchhoff stress (1PK) P of a given elastic energy density 
    e by 
    
        P_ij = d e(F) / d F_ij
    
    where F_ij is the entry at the i-th row and j-th column of the deformation 
    gradient and we assume that the energy density is a function of F only (no 
    other strain/deformation measures).

    Parameters
    ---------- 
    eng_density : symfem.functions.MatrixFunction
        strain energy density.
    F : None or symfem.functions.MatrixFunction
        symbolic deformation gradient of shape (ndim,ndim). 
    ndim : None or int
        number of spatial dimensions. Only needed if the other two arguments 
        are None, otherwise ignored.

    Returns
    -------
    P : symfem.functions.MatrixFunction
        symbolic 1. Piola-Kirchhoff stress (1PK) of shape (ndim,ndim).

    """
    #
    if is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    #
    S = []
    for i in range(E.shape[0]):
        S.append([eng_density.diff(variable=E[i,0])])
    return MatrixFunction(S)

def calculate_2pk(eng_density: ScalarFunction,
                  F : Union[None,MatrixFunction] = None,
                  E : Union[None,MatrixFunction] = None,
                  ndim: Union[int,None] = 3, 
                  **kwargs: Any) -> MatrixFunction:
    """
    Returns 2. Piola-Kirchhoff stress (2PK) S of a given elastic energy density 
    e by 
    
        S_i = d e(E) / d E_i
    
    where E_i is the i-th entry of the Lagrangian strain E in Voigt notation 
    and we assume that the energy density is a function of E only (no other 
    strain/deformation measures).

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
        symbolic 2. Piola-Kirchhoff stress (2PK).

    """
    #
    if E is None and F is not None:
        E = lagrangian_strain(ndim=ndim,
                              F=F)
    #
    if E is None and ndim is None:
        raise ValueError("E and ndim are none. ",
                         "Either ndim must be an integer or E must be given.")
    #
    if E is None:
        E = generate_constMatrix(ncol=1,
                                 nrow=int((ndim**2 + ndim) /2),
                                 name="E")
    #
    if not is_voigt(M=E,ndim=ndim):
        E = to_voigt(E)
    #
    S = []
    for i in range(E.shape[0]):
        S.append([eng_density.diff(variable=E[i,0])])
    return MatrixFunction(S)