# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import symbols
from symfem.functions import MatrixFunction

def shear_viscosity(ndim: int) -> MatrixFunction:
    """
    shear viscosity tensor.

    Parameters
    ----------
    ndim : int
        number of dimensions

    Returns
    -------
    eta : symfem.functions.MatrixFunction
        viscosity tensor.
    """
    eta = symbols("eta")
    if ndim == 1:
        return MatrixFunction([[E]])
    elif ndim == 2:
        return eta*MatrixFunction([[1,0,0],
                                   [0,1,0],
                                   [0,0,2]])
    elif ndim == 3:
        return eta*MatrixFunction([[1,0,0,0,0,0],
                                   [0,1,0,0,0,0],
                                   [0,0,1,0,0,0],
                                   [0,0,0,2,0,0],
                                   [0,0,0,0,2,0],
                                   [0,0,0,0,0,2]])