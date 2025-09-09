# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union
from itertools import product

from sympy import symbols, Symbol
from symfem.functions import MatrixFunction

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