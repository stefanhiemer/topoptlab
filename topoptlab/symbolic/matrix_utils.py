# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List, Tuple, Union
from warnings import warn
from itertools import product
from math import floor,sqrt

from sympy import symbols, Symbol, Function, Inverse, Trace
from symfem.functions import ScalarFunction,VectorFunction,MatrixFunction

from topoptlab.symbolic.utils import is_equal

def generate_constMatrix(ncol: int, nrow: int, name: str,
                         symmetric: bool = False,
                         return_symbols: bool = False
                         ) -> Tuple[MatrixFunction, List]:
    """
    Generate matrix full of symbolic constants.

    Parameters
    ----------
    ncol : int
        number of columns.
    nrow : int
        number of rows.
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
    
def generate_FunctMatrix(ncol: int, nrow: int, name: str,
                         variables: List,
                         symmetric: bool = False,
                         return_symbols: bool = False
                         ) -> Tuple[MatrixFunction, List]:
    """
    Generate matrix where each entry is a function of the variables.

    Parameters
    ----------
    ncol : int
        number of columns.
    nrow : int
        number of rows.
    name : str
        name to put in front of indices.
    variables : list
        list of symbols on which the matrix entries depend.
    symmetric : bool
        if True, matrix generated in symmetric fashion
    return_symbols : bool
        if True, returns the list of symbols of the entries. May be needed if
        you want to apply assumptions on them in a Sympy assumptions context.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with symbolic functions e. g. for a 2x2 
        matrix depending on x   [[M11(x),M12(x)],[M21(x),M22(x)]].
    symbol_list : list
        list of all symbols used for the matrix/vector

    """
    #
    M = []
    if return_symbols:
        function_list = []
    #
    for i in range(1,nrow+1):
        # create the matrix as list of lists
        if nrow != 1 and ncol !=1:
            functions = [name+str(i)+str(j) for j in range(1,ncol+1)]
        # create row vector as list
        elif nrow == 1:
            functions = [name+str(j) for j in range(1,ncol+1)]
        # create column vector as list
        elif ncol == 1:
            functions = [name+str(i) for j in range(1,ncol+1)]
        # introduce dependence on the variables
        functions = [Function(f)(*variables) for f in functions]
        #
        M.append( functions )
        if return_symbols:
            function_list = function_list + functions
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
        return MatrixFunction(M), function_list

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

def diag(v: Union[List,MatrixFunction]) -> Union[List,MatrixFunction]:
    """
    Return diagonal matrix with given diagonal values or extract diagonal
    values and return as list.

    Parameters
    ----------
    v : list or symfem.functions.MatrixFunction
        values of diagonal or matrix that needs diagonal extracted.
        
    Returns
    -------
    diag : list or symfem.functions.MatrixFunction
        diagonal matrix or extracted diagonal as list.
    """
    #
    if isinstance(v, list):
        size=len(v) 
        return MatrixFunction([[0 if i!=j else v[i] for j in range(size)] \
                                for i in range(size)])
    elif isinstance(v, MatrixFunction):
        
        return [v[i,i] for i in range(min(v.shape))]
    
def trace(M: MatrixFunction) -> ScalarFunction:
    """
    Take the trace of the given matrix `M`.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix M.
        
    Returns
    -------
    trace : symfem.functions.ScalarFunction
        trace of M.
    """ 
    return ScalarFunction(Trace(M.as_sympy()).simplify())

def to_square(v: MatrixFunction, order: str = "F") -> MatrixFunction:
    """
    Convert MatrixFunction of shape (n**2,1) to (n,n) either in 'C' or 'F'
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

def to_column(M: MatrixFunction, order: str = "C") -> MatrixFunction:
    """
    Flatten MatrixFunction of shape (m,n) to (m*n,1) either in 'C' or 'F'
    order analogously to numpy reshaping.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix of shape (m,n).
    order : str
        order of reshaping

    Returns
    -------
    v : symfem.functions.MatrixFunction
        matrix reshaped to shape (m*n,1)
    """
    m,n = M.shape
    v = []
    if order == "F":
        v = [[M[i%m, floor(i/n)]] for i in range(m*n)]
    elif order == "C":
        v = [[M[floor(i/n),i%m]] for i in range(m*n)]
    return MatrixFunction(v)

def kron(A: MatrixFunction,
         B: MatrixFunction) -> MatrixFunction:
    """
    Apply Kronecker product.

    Parameters
    ----------
    A : symfem.functions.MatrixFunction
        matrix of shape (m,n). 
    B : symfem.functions.MatrixFunction
        matrix of shape (p,q).
        
    Returns
    -------
    M : symfem.functions.MatrixFunction
        Kronecker product of shape (m*p,n*q)
    """
    m,n = A.shape
    p,q = B.shape
    M = []
    for i in range(m*p):
        M.append([])
        for j in range(n*q):
            M[-1].append(A[floor(i/p),floor(j/q)]*B[i%p,j%q])
    return MatrixFunction(M)

def eig(M: MatrixFunction) -> MatrixFunction:
    """
    Calculate eigenvalues and eigenvectors.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix of shape (n,n). 
        
    Returns
    -------
    eigenvalues : symfem.functions.MatrixFunction
        The eigenvalues repeated according to multiplicity, but not sorted 
        according to magnitude as symbolically this would need information 
        about the variables. Shape (n,1)
    eigenvectors : symfem.functions.MatrixFunction
        The normlaized eigenvectors such that the i-th column belongs to the 
        i-th eigenvalue. Shape (n,n)
    """
    #
    if not is_square(M):
        raise ValueError("M must be square to allow eigendecomposition. ",
                         "Current shape: ", M.shape)
    # use sympy for decomposition. 
    eigenvectors,eigenvalues = M.as_sympy().diagonalize(normalize=True)
    return diag(MatrixFunction(eigenvalues)),MatrixFunction(eigenvectors)

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
    size = A.shape[0]
    if size > 3:
        raise warn("This function is intended for size(M) < 4. ",
                   "Might get very slow otherwise.")
    if Adet is None:
        Adet = A.det()
    # 
    if size < 4:
        Ainv = [[[] for A in range(size)] for A in range(size)]
    if size == 1:
        Ainv[0][0] = 1 / Adet
    elif size == 2:
        Ainv[0][0], Ainv[1][1] = A[1][1]/Adet, A[0][0]/Adet
        Ainv[0][1], Ainv[1][0] = -A[0][1]/Adet, -A[1][0]/Adet
    elif size == 3:
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
    else:
        Ainv = Inverse(A.as_sympy()).simplify()
    return MatrixFunction(Ainv)

def det(M: MatrixFunction) -> MatrixFunction:
    """
    Calculate determinant.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix. 

    Returns
    -------
    det : symfem.functions.ScalarFunction
        determinant
    """
    return M.det()

def mul(M: MatrixFunction,
        s: Union[float,int,]) -> MatrixFunction:
    """
    Multiply each entry with scalar.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix. 

    Returns
    -------
    M : symfem.functions.ScalarFunction
        determinant
    """
    return M.det()

def from_voigt(M_v: MatrixFunction) -> MatrixFunction:
    """
    Convert MatrixFunction M_v of shape (d*(d+1)/2,1) from Voigt notation to 
    standard matrix notation M resulting in shape (d,d). M is square and 
    symmetric.

    Parameters
    ----------
    M_v : symfem.functions.MatrixFunction
        matrix in Voigt notation of shape (d*(d+1)/2,1). 

    Returns
    -------
    M : symfem.functions.MatrixFunction
        matrix of shape (d,d)
    """
    #
    d = int((-1+sqrt(1+8*M_v.shape[0]))/2)
    #
    M = [[0 for j in range(d)] for i in range(d)]
    # diagonal
    for i in range(d):
        M[i][i] = M_v[i,0]
    # off-diagonal
    if d == 2:
        M[0][1] = M_v[-1,0]
        M[1][0] = M_v[-1,0]
    elif d == 3:
        M[1][2] = M_v[-3,0]
        M[2][1] = M_v[-3,0]
        M[0][2] = M_v[-2,0]
        M[2][0] = M_v[-2,0]
        M[0][1] = M_v[-1,0]
        M[1][0] = M_v[-1,0]
    return MatrixFunction(M)

def to_voigt(M: MatrixFunction) -> MatrixFunction:
    """
    Convert MatrixFunction M of shape (d,d) to Voigt notation resulting in 
    shape (d*(d+1)/2,1). M must be square and symmetric.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix of shape (d,d). 

    Returns
    -------
    M_v : symfem.functions.MatrixFunction
        Voigt vector reshaped to shape (d*(d+1)/2,1)
    """
    #
    d = M.shape[0]
    #
    if d!=M.shape[1]:
        raise ValueError("M must be symmetric: ",M.shape)
    # diagonal
    M_v = [ [M[i,i]] for i in range(d)]
    #
    if d == 2:
        M_v.append( [M[0,1]] )
    elif d == 3:
        M_v.append([M[1,2]])
        M_v.append([M[0,2]]) 
        M_v.append([M[0,1]])
    return MatrixFunction(M_v)

def is_voigt(M: MatrixFunction, 
             ndim: int) -> bool:
    """
    Check if MatrixFunction M is notated in Voigt notation.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix function. 

    Returns
    -------
    is_voigt : bool
        True, if M is notated in Voigt notation.
    """
    if (M.shape [1]== 1) or (M.shape[0] == M.shape[1]): 
        return int(ndim*(ndim+1)/2)==M.shape[0]
    else:
        raise ValueError("This shape cannot be a common tensor in Voigt form.")
        
def is_square(M: MatrixFunction) -> bool:
    """
    Check if MatrixFunction M is a square matrix.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix function. 

    Returns
    -------
    is_square : bool
        True, if M is a square matrixn.
    """
    return M.shape[0] == M.shape[1]

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

def from_vectorfunction(vectorfunc : VectorFunction) -> MatrixFunction:
    """
    From VectorFunction of length v generate MatrixFunction of shape (v,1) by 
    storing values in a list of lists which is then converted to a 
    MatrixFunction.

    Parameters
    ----------
    vectorfunc : symfem.functions.VectorFunction
        Vectorfunction of length v.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        MatrixFunction of shape (v,1).

    """
    return MatrixFunction([[element] for element in vectorfunc]) 