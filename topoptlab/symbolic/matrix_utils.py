# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union
from itertools import product
from math import sqrt

from sympy import symbols, Symbol
from symfem.functions import ScalarFunction,MatrixFunction

from topoptlab.symbolic.utils import is_equal

def generate_constMatrix(ncol: int, nrow: int, name: str,
                         symmetric: bool = False,
                         return_symbols: bool = False
                         ) -> Tuple[MatrixFunction, List]:
    """
    Generate matrix full of symbolic constants as a list of lists.

    Parameters
    ----------
    ncol : int
        number of rows.
    nrow : int
        number of cols.
    name : str
        name to put in front of indices.
    symmetric : bool
        if True, matrix generated in symmetric fashion
    return_symbols : bool
        if True, returns the list of symbols of the entries. May be needed if
        you want to apply assumptions on them in a Sympy assumptions context.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with symbolic entries e. g. for a 2x2
        [[M11,M12],[M21,M22]].
    symbol_list : list
        list of all symbols used for the matrix/vector

    """
    #
    M = []
    if return_symbols:
        symbol_list = []
    for i in range(1,nrow+1):
        # create the matrix as list of lists
        if nrow != 1 and ncol !=1:
            variables = " ".join([name+str(i)+str(j) for j in range(1,ncol+1)])
        # create row vector as list
        elif nrow == 1:
            variables = " ".join([name+str(j) for j in range(1,ncol+1)])
        # create column vector as list
        elif ncol == 1:
            variables = " ".join([name+str(i) for j in range(1,ncol+1)])
        # create symbols
        variables = symbols(variables)
        # convert to list
        if isinstance(variables, Symbol):
            variables = [variables]
        elif isinstance(variables,tuple):
            variables = list(variables)
        #
        M.append( variables )
        if return_symbols:
            symbol_list = symbol_list + variables
    # make matrix symmetric
    if symmetric:
        if ncol != nrow:
            raise ValueError("For matrix to be symmetric, it must be square. Currenly ncol != nrow")
        #
        for i in range(nrow):
            for j in range(i+1,nrow):
                M[j][i] = M[i][j]
    #
    if not return_symbols:
        return MatrixFunction(M)
    else:
        return MatrixFunction(M), symbol_list

def simplify_matrix(M: Union[List,MatrixFunction]) -> MatrixFunction:
    """
    simplify element-wise the given MatrixFunction.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction or list
        matrix to be simplified.

    Returns
    -------
    M_new : symfem.functions.MatrixFunction
        simplified matrix.
    """
    if isinstance(M, MatrixFunction):
        nrow,ncol = M.shape[0],M.shape[1]
    elif isinstance(M, list):
        nrow,ncol = len(M),len(M[0])
    M_new = [[0 for j in range(ncol)] for i in range(nrow)]
    for i,j in product(range(M.shape[0]),range(M.shape[1])):
        M_new[i][j] = M[i,j].as_sympy().simplify()
    return MatrixFunction(M_new)

def eye(size: int) -> MatrixFunction:
    """
    Return identity matrix of given size.

    Parameters
    ----------
    size : int
        size of identity matrix.

    Returns
    -------
    eye : symfem.functions.MatrixFunction
        identity matrix of shape (size,size).
    """
    
    return MatrixFunction([[0 if i!=j else 1 for j in range(size)] \
                          for i in range(size)])

def diag(vals: List) -> MatrixFunction:
    """
    Return diagonal matrix with given diagonal values.

    Parameters
    ----------
    vals : list
        values of diagonal.

    Returns
    -------
    diag : symfem.functions.MatrixFunction
        diagonal matrix.
    """
    size=len(vals)
    return MatrixFunction([[0 if i!=j else vals[i] for j in range(size)] \
                          for i in range(size)])

def to_square(v, order: str = "C") -> MatrixFunction:
    """
    Return MatrixFunction of shape (n**2,1) reshape (n,n) either in 'C' or 'F'
    order analogously to numpy reshaping.

    Parameters
    ----------
    v : symfem.functions.MatrixFunction
        matrix of shape (n**2,1).
    order : str
        order of reshaping

    Returns
    -------
    M : symfem.functions.MatrixFunction
        matrix reshaped to shape (n,n)
    """
    n = int( sqrt(v.shape[0]) )
    M = []
    if order == "F":
        for i in range(n):
            M.append([v[j*n + i,0] for j in range(n)])
    elif order == "C":
        for i in range(n):
            M.append([v[i*n+j,0] for j in range(n)])
    return MatrixFunction(M)

def inverse(A: MatrixFunction,
            Adet: Union[None, ScalarFunction] = None) -> MatrixFunction:
    """
    Return the symbolic inverse of a square matrix using explicit adjugate 
    formulas. Does not work if dim(A) > 3.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
       square matrix of shape (ndim, ndim).
   Adet : symfem.functions.ScalarFunction or None, optional
       determinant of ``A``. Providing it allows reuse.
       
    Returns
    -------
    Ainv : symfem.functions.MatrixFunction
       Matrix inverse of shape (ndim, ndim).

    """
    #
    ndim = A.shape[0]
    if ndim > 3:
        raise ValueError("This function only works for square matrices ndim < 4.")
    if Adet is None:
        Adet = A.det()
    # adAungate matrix
    Ainv = [[[] for A in range(ndim)] for A in range(ndim)]
    if ndim == 1:
        Ainv[0][0] = 1 / Adet
    elif ndim == 2:
        Ainv[0][0], Ainv[1][1] = A[1][1]/Adet, A[0][0]/Adet
        Ainv[0][1], Ainv[1][0] = -A[0][1]/Adet, -A[1][0]/Adet
    elif ndim == 3:
        #
        Ainv[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) / Adet
        Ainv[0][1] = -(A[0][1]*A[2][2] - A[0][2]*A[2][1]) / Adet
        Ainv[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / Adet
        #
        Ainv[1][0] = -(A[1][0]*A[2][2] - A[1][2]*A[2][0]) / Adet
        Ainv[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0] ) / Adet
        Ainv[1][2] = -(A[0][0]*A[1][2] - A[0][2]*A[1][0]) / Adet
        #
        Ainv[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) / Adet
        Ainv[2][1] = -(A[0][0]*A[2][1] - A[0][1]*A[2][0]) / Adet
        Ainv[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) / Adet
    return MatrixFunction(Ainv)

def trace(A: MatrixFunction) -> ScalarFunction:
    """
    Return trace of square matrix.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
       square matrix of shape (ndim, ndim).
       
    Returns
    -------
    trace : symfem.functions.ScalarFunction
       trace of A.

    """
    #
    check_square(A)
    #
    trace = 0
    for i in range(A.shape[0]):
        trace = trace + A[i,i]
    return trace

def check_square(A: MatrixFunction) -> None:
    """
    Check if A is a square matrix.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
       matrix to be checked.
       
    Returns
    -------
    None

    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("A is not square: ", A.shape)
    return

def matrix_equal(A: MatrixFunction, B: MatrixFunction) -> bool:
    """
    Check if matrix A is equal to matrix B.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
       matrix to be checked.
    B : symfem.functions.MatrixFunction
       matrix to be checked.
       
    Returns
    -------
    is_equal
       True if A==B.     

    """
    
    rows,cols = A.shape 
    # check shape identical
    if A.shape!=B.shape:
        return False
    # check entries
    for i in range(rows):
        for j in range(cols):
            if not is_equal(A[i,j],B[i,j]):
                return False
    return True