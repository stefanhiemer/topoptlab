# SPDX-License-Identifier: GPL-3.0-or-later
import math

from symfem.functions import MatrixFunction

def convert_to_voigt(A: MatrixFunction) -> MatrixFunction:
    """
    Convert 2nd rank tensor into its Voigt representation.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction, shape (ndim,ndim)
        2nd rank tensor

    Returns
    -------
    A_v : symfem.functions.MatrixFunction, shape ((ndim**2 + ndim) /2, 1)
        2nd rank tensor in voigt notation
    """
    #
    if isinstance(A,MatrixFunction):
        ndim = A.shape[0]
    elif isinstance(A,list):
        ndim = len(A)
    #
    A_v = [[A[i][i]] for i in range(ndim)]
    if ndim == 3:
        A_v += [[A[1][-1]]]
    A_v += [[A[0][i]] for i in range(ndim-1,0,-1)]
    return MatrixFunction(A_v)

def convert_from_voigt(A_v: MatrixFunction) -> MatrixFunction:
    """
    Convert 2nd rank tensor into from its Voigt representation to the standard
    matrix represenation.

    Parameters
    ----------
    A_v : symfem.functions.MatrixFunction, shape ((ndim**2 + ndim) /2, 1)
        2nd rank tensor in Voigt represenation (so a column vector)

    Returns
    -------
    A : symfem.functions.MatrixFunction, shape (ndim,ndim)
        2nd rank tensor in matrix notation
    """
    #
    if isinstance(A_v,MatrixFunction):
        l = A_v.shape[0]
    elif isinstance(A_v,list):
        l = len(A_v)
    #
    if l not in [1,3,6]:
        raise ValueError("This is not a vector compatible with the assumptions of Voigt representation.")
    #
    ndim = int( -1/2 + math.sqrt(1/2+2*l) )
    #
    A = [[0 for j in range(ndim)] for i in range(ndim)]
    for i in range(ndim):
        A[i][i] = A_v[i][0]
    if ndim > 1:
        A[0][1] = A_v[-1][0]
        A[1][0] = A_v[-1][0]
    if ndim > 2:
        A[1][2] =  A_v[3][0]
        A[2][1] =  A_v[3][0]
        A[0][2] =  A_v[4][0]
        A[2][0] =  A_v[4][0]
    return MatrixFunction(A)
