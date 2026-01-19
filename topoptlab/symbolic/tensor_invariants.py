# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union
from itertools import product
from math import sqrt

from sympy import symbols, Symbol
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import check_square, diag, trace

def principal_invariants(A: MatrixFunction) -> Tuple:
    """
    Return the principal tensor invariants `I` of MatrixFunction `A`. Works 
    only for dim(A) <= 3.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
        square matrix.

    Returns
    -------
    I1 : symfem.functions.ScalarFunction
        first principal invariant (trace).
    I2 : symfem.functions.ScalarFunction
        second principal invariant.
    I3 : symfem.functions.ScalarFunction
        third principal invariant (determinant).
    """
    check_square(A)
    ndim = A.shape[0]

    if ndim == 1:
        return A[0, 0]
    elif ndim == 2:
        return trace(A), A.det()
    elif ndim == 3:
        I1 = trace(A)
        I2 = (I1**2 - trace(A @ A)) / 2
        I3 = A.det()
        return I1, I2, I3
    else:
        raise ValueError("This function only works for square matrices with dim ≤ 3.")


def main_invariants(A: MatrixFunction) -> Tuple:
    """
    Return the main tensor invariants `I` of MatrixFunction `A` expressed in 
    terms of the principal invariants. Works only for dim(A) <= 3.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
        square matrix.

    Returns
    -------
    J1 : symfem.functions.ScalarFunction
        first main invariant.
    J2 : symfem.functions.ScalarFunction
        second main invariant.
    J3 : symfem.functions.ScalarFunction
        third main invariant.
    """
    check_square(A)
    ndim = A.shape[0]

    if ndim == 1:
        return A[0, 0]
    elif ndim == 2:
        return trace(A), A.det()
    elif ndim == 3:
        I1, I2, I3 = principal_invariants(A)
        J1 = I1
        J2 = I1**2 - 2*I2
        J3 = I1**3 - 3*I1*I2 + 3*I3
        return J1, J2, J3
    else:
        raise ValueError("This function only works for square matrices with dim ≤ 3.")
