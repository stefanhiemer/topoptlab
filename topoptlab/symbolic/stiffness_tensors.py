# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols
from symfem.functions import MatrixFunction

def stifftens_isotropic(ndim: int, 
                        plane_stress: bool = True) -> MatrixFunction:
    """
    stiffness tensor for isotropic material expressed in Terms of Young's
    modulus `E` and Poisson's ratio ´nu`.

    Parameters
    ----------
    ndim : int
        number of dimensions
    plane_stress : bool
        if True, return stiffness tensor for plane stress, otherwise return
        stiffness tensor for plane strain

    Returns
    -------
    c : symfem.functions.MatrixFunction
        stiffness tensor.
    """
    E,nu = symbols("E nu")
    if ndim == 1:
        return MatrixFunction([[E]])
    elif ndim == 2:
        if plane_stress:
            return E/(1-nu**2)*MatrixFunction([[1,nu,0],
                                               [nu,1,0],
                                               [0,0,(1-nu)/2]])
        else:
            return E/((1+nu)*(1-2*nu))*MatrixFunction([[1-nu,nu,0],
                                                       [nu,1-nu,0],
                                                       [0,0,(1-nu)/2]])
    elif ndim == 3:
        return E/((1+nu)*(1-2*nu))*MatrixFunction([[1-nu,nu,nu,0,0,0],
                                                   [nu,1-nu,nu,0,0,0],
                                                   [nu,nu,1-nu,0,0,0],
                                                   [0,0,0,(1-nu)/2,0,0],
                                                   [0,0,0,0,(1-nu)/2,0],
                                                   [0,0,0,0,0,(1-nu)/2]])

