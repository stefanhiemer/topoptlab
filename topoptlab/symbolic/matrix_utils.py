# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Tuple, Union
from warnings import warn
from itertools import product
from math import floor,sqrt
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sympy import symbols, Symbol, Function, Inverse, Trace,\
                  cse, factor, factor_terms, gcd, sympify, gcd_terms, Mul, \
                  Integer, Float
from sympy.core.expr import Expr
from symfem.references import Reference
from symfem.functions import ScalarFunction,VectorFunction,MatrixFunction
from symfem.symbols import AxisVariablesNotSingle

from topoptlab.symbolic.utils import is_equal, take_generic_branch

def generate_constMatrix(ncol: int, nrow: int, name: str,
                         symmetric: bool = False,
                         return_symbols: bool = False, 
                         assumptions: Dict = {},
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
        variables = symbols(variables, **assumptions)
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

def flatten(M: MatrixFunction, order: str = "C") -> VectorFunction:
    """
    Flatten MatrixFunction of shape (m,n) to VectorFunction (m*n) either in 
    'C' or 'F' order analogously to numpy flatten().

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix of shape (m,n).
    order : str
        order of reshaping

    Returns
    -------
    v : symfem.functions.VectorFunction
        flattened matrix to VectorFunction shape (m*n)
    """
    m,n = M.shape
    v = []
    if order == "F":
        v = [M[i%m, floor(i/m)] for i in range(m*n)]
    elif order == "C":
        v = [M[floor(i/n),i%n] for i in range(m*n)]
    return VectorFunction(v)

def _operation_on_entry(args: Tuple) -> Tuple:
    """
    Helper for apply_elementwise().

    Parameters
    ----------
    args : tuple
        (expr, operation, func_args)

    Returns
    -------
    tuple
        result
    """
    expr, operation, func_args = args
    return operation(expr, **func_args)

def _apply_elementwise(M: Union[VectorFunction,MatrixFunction], 
                      func : Callable,
                      func_args : Dict,
                      dscr : str,
                      return_symfemfunc : bool = True,
                      parallel: Union[None,bool] = None, 
                      parallel_threshold : int = 16,
                      symmetry : Union[None,bool] = None
                      ) -> Union[MatrixFunction,List]:
    """
    Apply symbolic operation to a VectorFunction or MatrixFunction 
    element-wise. For each entry, a task is created, dispatched to func and
    then collected with which ´M´ is then reconstructed.
    
    This function allows parallel execution via the multiprocessing package.
    
    Parameters
    ----------
    M : symfem.functions.VectorFunction or symfem.functions.MatrixFunction.
        vector/matrix to which operation to be applied to.
    func : callable
        function applied to each element.
    func_args : dict
        dictionary with keys identical to function arguments.
    dscr : str 
        description of function displayed on the progress bar.
    return_symfemfunc : bool
        if True, returns MatrixFunction which is basically in same spirit as 
        element-wise operations in numpy. If False returns a list of lists 
        with identical shapes as the MatrixFunction. Needed if func returns 
        more than one output.
    parallel : None or bool
        if True, operation is parallelized via Pool. If None, autmatically 
        switch to parallelization if entries are larger than parallel_threshold.
    parellel_threshold : int 
        if parallel is None and if size(M) > parallel_threshold, 
        parallelization is used.  
    symmetry : None or bool
        if True, assume M to be symmetric and reduce number of function calls 
        accordingly and copies entries afterwards. Only applies to square 
        matrices.
        
    Returns
    -------
    M_applied : symfem.functions.VectorFunction symfem.functions.MatrixFunction
        element-wise operated on vector/matrix.  
    
    """
    # avoid repetitive instance check
    ismatrixfunc = isinstance(M, MatrixFunction)
    isvectorfunc = isinstance(M,VectorFunction)
    #
    if ismatrixfunc:
        nrow,ncol = M.shape[0],M.shape[1]
        size = nrow*ncol
    elif isvectorfunc:
        nrow = M.shape[0]
        size=nrow
    else:
        raise TypeError("M must be either symfem.functions.MatrixFunction ", 
                        "or symfem.functions.VectorFunction. Current type: ", 
                        type(M))
    # symmetry only valid for square matrices
    if ismatrixfunc and not is_square(M) and symmetry:
        warn("symmetry only applies to square matrices. Symmetry will be ignored.")
        symmetry = False
    # check if parallel execution
    if parallel is None and size > parallel_threshold:
        parallel = True
    # create tasks in a list or list-of-lists with C ordering
    if symmetry and ismatrixfunc:
        tasks = [(M[i][j], func, func_args)
                 for i in range(nrow) for j in range(i, ncol)]
    elif ismatrixfunc:
        tasks = [(M[i][j], func, func_args)
                 for i in range(nrow) for j in range(ncol)]
    elif isvectorfunc:
        tasks = [(M[i], func, func_args)
                 for i in range(nrow)]
    # serial
    if not parallel:
        M_new = [_operation_on_entry(task) for task in tasks]
    else:
        # integrate in parallel and then write to M_new
        with Pool(cpu_count()) as pool:
            M_new = list(tqdm(pool.imap(_operation_on_entry, 
                                        tasks),
                              total=len(tasks),
                              desc=dscr))
    # re-assemble
    if ismatrixfunc:
        if symmetry:
            M_new = [[M_new[i*ncol + j - (i+1)*i//2] if i<=j else 0 \
                      for j in range(ncol)] for i in range(nrow)]
            for i in range(nrow):
                for j in range(i+1,ncol):
                    M_new[j][i] = M_new[i][j]
        else:
            M_new = [[M_new[i*ncol + j] for j in range(ncol)] for i in range(nrow)]
    #
    if return_symfemfunc and ismatrixfunc:
        return MatrixFunction(M_new)
    elif return_symfemfunc and isvectorfunc:
        return VectorFunction(M_new)
    else:
        return M_new 

def _factor_expr(expression : Expr, 
                 variables : List) -> Expr:
    """
    Factorize sympy expression. Extract the factor containing not the listed 
    variables.

    Parameters
    ----------
    expression : sympy.Expr
        sympy expression.
    variables : list
        list of variables which should not appear in the factored out 
        expression.

    Returns
    -------
    factor: sympy.Expr
        factored out sympy expression.
    remainder: sympy.Expr
        remaining sympy expression.
    """
    #
    factor, remainder = factor_terms(expression).as_independent(*variables, 
                                                                as_Add=False)
    # unfortunately if there is no factor, sympy returns 0. exchange for 1 to 
    # avoid zero division
    if factor is None:
        factor = sympify(1)
    elif factor == 0:
        factor = sympify(1)
    return factor, remainder

def factor_matrix(M : Union[VectorFunction,MatrixFunction],
                  variables : List,
                  parallel : Union[None,bool] = None, 
                  symmetry : bool = False
                  ) -> Union[VectorFunction,MatrixFunction]:
    """
    Factor element-wise the given MatrixFunction by first pull out factors not 
    including the variables, find the greatest common divisor and 
    extract it from the matrix. 

    Parameters
    ----------
    M : symfem.functions.VectorFunction or symfem.functions.MatrixFunction
        vector/matrix to be simplified.
    variables : list
        list of variables which should not appear in the factored out 
        expression.
    parallel : None or bool
        if True, simplification is parallelized via Pool. If None, autmatically 
        switch to parallelization if entries are larger than 16.
    symmetry : None or bool
        if True, assume M to be symmetric and reduce number of function calls 
        accordingly and copies entries afterwards. Only applies to square 
        matrices.

    Returns
    -------
    remainder : symfem.functions.MatrixFunction
        factored-out matrix.
    factor : sympy.Expr
        greatest common divisor 
    """
    # avoid repetitive instance check
    ismatrixfunc = isinstance(M, MatrixFunction)
    isvectorfunc = isinstance(M,VectorFunction)
    #
    if ismatrixfunc:
        nrow,ncol = M.shape[0],M.shape[1]
        size = nrow*ncol
    elif isvectorfunc:
        nrow = M.shape[0]
        size=nrow
    else:
        raise TypeError("M must be either symfem.functions.MatrixFunction ", 
                        "or symfem.functions.VectorFunction. Current type: ", 
                        type(M))
    #
    if ismatrixfunc:
        M = flatten(to_column(M=M,order="C"))
    #
    split = _apply_elementwise(M=M, 
                               func = _factor_expr,
                               dscr = "Factoring out MatrixFunction",
                               return_symfemfunc=False,
                               func_args = {"variables": variables},
                               parallel = parallel, 
                               symmetry = symmetry)
    # extract factors/remainders 
    factors = [split[i][0] for i in range(size)]
    remainders = [split[i][1] for i in range(size)]
    # 2- compare factors and extract the greatest common divisor (gcd)
    #common_fac = gcd_terms(Mul(*factors))
    g = factor_terms(sum(factors))
    common_fac, _ = g.as_independent(*variables, 
                                     as_Add=False)
    if common_fac == 0:
        common_fac = Integer(1)
    # 3.do factors/common_factor and multiply this to the remainders 
    M_new = [r*(f/common_fac) for f,r in zip(factors,remainders)]
    # assemble simplified matrix
    if ismatrixfunc:
        M_new = [[M_new[i*nrow + j] for j in range(ncol)] for i in range(nrow)] 
        return MatrixFunction(M_new), common_fac
    elif isvectorfunc:
        return VectorFunction(M_new), common_fac

def _simplify_expr(expression: ScalarFunction) -> ScalarFunction:
    """
    Simplify sympy expression.

    Parameters
    ----------
    expression : ScalarFunction
        sympy expression.

    Returns
    -------
    expression_simplified : ScalarFunction
        simplified ScalarFunction.

    """
    if isinstance(expression, ScalarFunction):
        return ScalarFunction(expression.as_sympy().simplify())
    elif isinstance(expression,bool) and not expression:
        return False
    else: 
        raise TypeError("Wrong type for simplification: ",
                        type(expression))

def simplify_matrix(M: Union[VectorFunction,MatrixFunction], 
                    parallel: Union[None,bool] = None, 
                    eliminate_piecewise: bool = False, 
                    symmetry : bool = False
                    ) -> Union[VectorFunction,MatrixFunction]:
    """
    Simplify element-wise the given MatrixFunction by first remove common 
    factors (sympy.factor_terms), simplification of the remaining expression 
    and subsequent merging. 

    Parameters
    ----------
    M : symfem.functions.VectorFunction or symfem.functions.MatrixFunction
        matrix to be simplified.
    parallel : None or bool
        if True, simplification is parallelized via Pool. If None, autmatically 
        switch to parallelization if entries are larger than 16.
    eliminate_piecewise : bool 
        if True, eliminate piecewise definitions heuristically by taking the  
        branch with the longest condition in terms of its string representation
        (uses symbolic.utils.take_generic_branch).
    symmetry : None or bool
        if True, assume M to be symmetric and reduce number of function calls 
        accordingly and copies entries afterwards. Only applies to square 
        matrices.
        
    Returns
    -------
    M_new : symfem.functions.VectorFunction or symfem.functions.MatrixFunction
        simplified vector/matrix.
    """
    #
    M_new = _apply_elementwise(M=M, 
                               func = _simplify_expr,
                               dscr = "Simplifying MatrixFunction.",
                               return_symfemfunc = True,
                               func_args = {},
                               parallel = parallel, 
                               symmetry = symmetry)
    #
    if eliminate_piecewise:
        M_new = _apply_elementwise(M=M, 
                                   func = take_generic_branch,
                                   dscr = "Take general branch for piecewise expressions.",
                                   return_symfemfunc = True,
                                   func_args = {},
                                   parallel = parallel, 
                                   symmetry = symmetry)
    # merge factored and reduced 
    #exprs = [f * r for f, r in zip(common_factors, reduced)]
    # assemble simplified matrix
    return M_new

def _integrate_entry(f : ScalarFunction, 
                     domain: Reference,
                     variables: AxisVariablesNotSingle,
                     dummy_vars: AxisVariablesNotSingle, ) -> Tuple:
    """
    Helper function for integrate() that computes the integral of the element 
    M[i,j] of MatrixFunction M.
    
    Parameters
    ----------
    f : symfem.functions.ScalarFunction
        entry to be integrated.
    domain : symfem.references.Reference
        domain of the integral
    vars : symfem.symbols.AxisVariablesNotSingle
        variables to integrate with respect to.
    dummy_vars :  symfem.symbols.AxisVariablesNotSingle
        dummy variables to use inside the integral.

    Returns
    -------
    integrated : tuple
        tuple that contains i, j, M_integrated[i,j], domain, vars, and 
        dummy_vars.  
    
    """
    return f.integral(domain, variables, dummy_vars)
    

def integrate(M: Union[VectorFunction,MatrixFunction], 
              domain: Reference,
              variables: AxisVariablesNotSingle,
              dummy_vars: AxisVariablesNotSingle, 
              parallel: Union[None,bool] = None, 
              symmetry : Union[None,bool] = None)-> MatrixFunction:
    """
    Compute the integral of the VectorFunction or MatrixFunction element-wise .
    
    Parameters
    ----------
    M : symfem.functions.MatrixFunction or symfem.functions.MatrixFunction.
        vector or matrix to be integrated.
    domain : symfem.references.Reference
        domain of the integral
    vars : symfem.symbols.AxisVariablesNotSingle
        variables to integrate with respect to.
    dummy_vars :  symfem.symbols.AxisVariablesNotSingle
        dummy variables to use inside the integral.
    parallel : None or bool
        if True, simplification is parallelized via Pool. If None, autmatically 
        switch to parallelization if entries are larger than 16
    symmetry : None or bool
        if True, assume M to be symmetric and reduce number of integrations 
        accordingly and copies entries afterwards. Only applies to square 
        matrices.
        
    Returns
    -------
    M_integrated : symfem.functions.MatrixFunction
        element-wise integrated vector/matrix.  
    
    """
    return _apply_elementwise(M=M, 
                              func = _integrate_entry,
                              dscr = "Integration MatrixFunction.",
                              return_symfemfunc = True,
                              func_args = {"domain": domain, 
                                           "variables": variables, 
                                           "dummy_vars": dummy_vars},
                              parallel = parallel, 
                              symmetry = symmetry)

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
    
def trace(M: MatrixFunction,
          mode: str="symfem") -> Union[ScalarFunction]:
    """
    Take the trace of the given matrix `M`.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix M.
    mode : str
        if 'symfem' returns symfem.functions.ScalarFunction, else 
        returns sympy expression
        
    Returns
    -------
    trace : sympy.Expr or symfem.functions.ScalarFunction
        trace of M.
    """ 
    if mode=="symfem":
        return ScalarFunction(Trace(M.as_sympy()).simplify())
    else:
        return Trace(M.as_sympy()).simplify()

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
    if not all([isinstance(A, MatrixFunction),
                isinstance(B,MatrixFunction)]):
        raise TypeError("A and B must be symfem.functions.MatrixFunction. ",
                        "Current type A: ", type(A), 
                        "Current type B: ", type(B))
    #
    m,n = A.shape
    p,q = B.shape
    #
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
        s: Union[float,int]) -> MatrixFunction:
    """
    Multiply each entry with scalar.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix. 
    s : float or int
        scalar
    
    Returns
    -------
    M_new : symfem.functions.MatrixFunction
        MatrixFunction multiplied with s.
    """
    if isinstance(s, int):
        return M.__mul__(Integer(s))
    elif isinstance(s, float): 
        return M.__mul__(Float(s))

def from_voigt(M_v: MatrixFunction, 
               eng_conv: bool = False) -> MatrixFunction:
    """
    Convert MatrixFunction M_v of shape (d*(d+1)/2,1) from Voigt notation to 
    standard matrix notation M resulting in shape (d,d). M is square and 
    symmetric. Applies only to rank 2 tensors right now.

    Parameters
    ----------
    M_v : symfem.functions.MatrixFunction
        matrix in Voigt notation of shape (d*(d+1)/2,1). 
    eng_conv : bool 
        if True, engineering convention is applied meaning the shear components
        are scaled by a factor of two in Voigt notation. Usually applies only 
        to strains.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        matrix of shape (d,d)
    """
    #
    ndim = int((-1+sqrt(1+8*M_v.shape[0]))/2)
    #
    M = [[0 for j in range(ndim)] for i in range(ndim)]
    # diagonal
    for i in range(ndim):
        M[i][i] = M_v[i,0]
    # off-diagonal
    if ndim == 2:
        M[0][1] = M_v[-1,0]
        M[1][0] = M_v[-1,0]
    elif ndim == 3:
        M[1][2] = M_v[-3,0]
        M[2][1] = M_v[-3,0]
        M[0][2] = M_v[-2,0]
        M[2][0] = M_v[-2,0]
        M[0][1] = M_v[-1,0]
        M[1][0] = M_v[-1,0]
    # revert the factor 2 scaling
    if eng_conv:
        for i in range(ndim):
            for j in range(i+1,ndim):
                M[i][j] = M[i][j]/2
                M[j][i] = M[j][i]/2
    return MatrixFunction(M)

def to_voigt(M: MatrixFunction, 
             eng_conv: bool = False) -> MatrixFunction:
    """
    Convert MatrixFunction M of shape (d,d) to Voigt notation resulting in 
    shape (d*(d+1)/2,1). M must be square and symmetric. Applies only to rank 2
    tensors right now. 

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix of shape (d,d). 
    eng_conv : bool 
        if True, engineering convention is applied meaning the shear components
        are scaled by a factor of two in Voigt notation. Usually applies only 
        to strains.

    Returns
    -------
    M_v : symfem.functions.MatrixFunction
        Voigt vector reshaped to shape (d*(d+1)/2,1)
    """
    #
    ndim = M.shape[0]
    #
    if ndim!=M.shape[1]:
        raise ValueError("M must be symmetric: ",M.shape)
    # diagonal
    M_v = [ [M[i,i]] for i in range(ndim)]
    #
    if ndim == 2:
        M_v.append( [M[0,1]] )
    elif ndim == 3:
        M_v.append([M[1,2]])
        M_v.append([M[0,2]]) 
        M_v.append([M[0,1]])
    # apply factor 2 scaling
    if eng_conv:
        for i in range(ndim,len(M_v)):
            M_v[i][0]=2*M_v[i][0]
    return MatrixFunction(M_v)

def is_voigt(M: MatrixFunction, 
             ndim: int) -> bool:
    """
    Check if MatrixFunction M is notated in Voigt notation. Applies only to 
    rank 2 tensors right now.

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