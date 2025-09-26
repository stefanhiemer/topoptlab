# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union
from itertools import product
from math import sqrt

from sympy import symbols, Symbol
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.matrix_utils import check_square, diag, trace

def main_invariants(A: MatrixFunction) -> Tuple:
    """
    Return main tensor invariants J of A. Does not work if dim(A) > 3.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
        square matrix.

    Returns
    -------
    J1 : symfem.functions.ScalarFunction
        first main invariant
    J2 : symfem.functions.ScalarFunction
        second main invariant
    J3 : symfem.functions.ScalarFunction
        third main invariant
    """
    #
    check_square(A)
    #
    ndim = A.shape[0]
    #
    if ndim == 1:
        return A
    elif ndim == 2:
        return trace(A),A.det()
    elif ndim == 3:
        I1,I2,I3 = principal_invariants
        return I1, I1**2 - 2*I2, I1**3 - 3*(I1*I2 + I3)
    else:
        raise ValueError("This function only works for square matrices ndim < 4.")

def principal_invariants(A: MatrixFunction) -> Tuple:
    """
    Return principal tensor invariants I of A. Does not work if dim(A) > 3.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
        square matrix.

    Returns
    -------
    I1 : symfem.functions.ScalarFunction
        first principal invariant
    I2 : symfem.functions.ScalarFunction
        second principal invariant
    I3 : symfem.functions.ScalarFunction
        third principal invariant
    """
    #
    check_square(A)
    #
    ndim = A.shape[0]
    #
    if ndim == 1:
        return A
    elif ndim == 2:
        return trace(A),A.det()
    elif ndim == 3:
        return trace(A), diag([1/2,1/2,1/2])@(trace(A)**2 - trace(A@A)), A.det()
    else:
        raise ValueError("This function only works for square matrices ndim < 4.")
    