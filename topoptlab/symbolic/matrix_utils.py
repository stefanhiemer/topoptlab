# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Callable
from typing import Any
from warnings import warn
from math import floor,sqrt
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sympy import symbols, Symbol, Function, Inverse, Trace,\
                  factor_terms, sympify, Integer, Float, expand,\
                  Rational, igcd, ilcm
from sympy.core.expr import Expr
from symfem.references import Reference
from symfem.functions import ScalarFunction,VectorFunction,MatrixFunction
from symfem.symbols import AxisVariablesNotSingle

from topoptlab.symbolic.utils import is_equal, take_generic_branch,\
                                    separate_terms
from topoptlab.symbolic.scalar_utils import integrate_expression

def generate_constMatrix(ncol: int, nrow: int, name: str,
                         symmetric: bool = False,
                         return_symbols: bool = False, 
                         assumptions: dict = {},
                         ) -> tuple[MatrixFunction, list]:
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

def full(ncol: int, nrow: int,
         fill_value: int | float,
         ) -> MatrixFunction:
    """
    Generate matrix with all entries equal to fill_value.

    Parameters
    ----------
    ncol : int
        number of columns.
    nrow : int
        number of rows.
    fill_value : int or float
        entry of each element

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with fill_value].

    """
    return MatrixFunction([[fill_value for _ in range(ncol)] for _ in range(nrow)])

def zeros(ncol: int, nrow: int) -> MatrixFunction:
    """
    Generate matrix of zeros.

    Parameters
    ----------
    ncol : int
        number of columns.
    nrow : int
        number of rows.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with zeros.
    """
    return full(ncol=ncol, nrow=nrow, fill_value=0)

def ones(ncol: int, nrow: int) -> MatrixFunction:
    """
    Generate matrix of ones.

    Parameters
    ----------
    ncol : int
        number of columns.
    nrow : int
        number of rows.

    Returns
    -------
    M : symfem.functions.MatrixFunction
        symfem MatrixFunction filled with ones.
    """
    return full(ncol=ncol, nrow=nrow, fill_value=1)

def generate_FunctMatrix(ncol: int, nrow: int, name: str,
                         variables: list,
                         symmetric: bool = False,
                         return_symbols: bool = False
                         ) -> tuple[MatrixFunction, list]:
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

def _operation_on_entry(args: tuple) -> tuple:
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

def _apply_elementwise(M: VectorFunction | MatrixFunction, 
                      func : Callable,
                      func_args : dict,
                      dscr : str,
                      return_symfemfunc : bool = True,
                      parallel: None | bool = None, 
                      parallel_threshold : int = 16,
                      symmetry : None | bool = None
                      ) -> MatrixFunction | list:
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
                 variables : list) -> Expr:
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

def factor_matrix(M : VectorFunction | MatrixFunction,
                  variables : list,
                  parallel : None | bool = None,
                  symmetry : bool = False
                  ) -> VectorFunction | MatrixFunction:
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

def common_factor(factors: list) -> Expr:
    """
    Greatest common factor of a list of products, e. g. the factors
    extracted by _factor_expr. A base is part of the common factor if it
    appears in every nonzero factor with exponents of the same sign; its
    exponent is the one with the smallest magnitude. Rational numbers
    contribute the gcd of their numerators divided by the lcm of their
    denominators.

    Parameters
    ----------
    factors : list
        list of sympy expressions (products).

    Returns
    -------
    common : sympy.Expr
        common factor. 1 if there is none.

    """
    factors = [sympify(f) for f in factors if sympify(f) != 0]
    if len(factors) == 0:
        return Integer(1)
    # numerical part
    numbers = [f.as_coeff_Mul()[0] for f in factors]
    common = Rational(igcd(*[abs(n.p) for n in numbers]),
                      ilcm(*[n.q for n in numbers]))
    # symbolic part
    powers = [f.as_coeff_Mul()[1].as_powers_dict() for f in factors]
    for base in powers[0]:
        exps = [sympify(p.get(base, 0)) for p in powers]
        if not all(e.is_number for e in exps):
            continue
        if all(e > 0 for e in exps):
            common *= base**min(exps)
        elif all(e < 0 for e in exps):
            common *= base**max(exps)
    return common

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

def simplify_matrix(M: VectorFunction | MatrixFunction, 
                    parallel: None | bool = None, 
                    eliminate_piecewise: bool = False, 
                    symmetry : bool = False
                    ) -> VectorFunction | MatrixFunction:
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
        M_new = _apply_elementwise(M=M_new, 
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
                     dummy_vars: AxisVariablesNotSingle,
                     backend: str = "sympy",
                     backend_kws: None | dict = None) -> ScalarFunction:
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
    backend : str
        integrator, either "sympy" or "symint". See
        topoptlab.symbolic.scalar_utils.integrate_expression.
    backend_kws : None or dict
        keyword arguments passed to symintegral.integrate. Only used for
        "symint".

    Returns
    -------
    integrated : symfem.functions.ScalarFunction
        integrated entry.
    
    """
    return ScalarFunction(integrate_expression(expression=f, domain=domain,
                                               variables=variables,
                                               dummy_vars=dummy_vars,
                                               backend=backend,
                                               backend_kws=backend_kws))
    

def integrate(M: VectorFunction | MatrixFunction, 
              domain: Reference,
              variables: AxisVariablesNotSingle,
              dummy_vars: AxisVariablesNotSingle, 
              parallel: None | bool = None, 
              symmetry : None | bool = None,
              backend: str = "sympy",
              backend_kws: None | dict = None)-> MatrixFunction:
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
    backend : str
        integrator, either "sympy" or "symint". See
        topoptlab.symbolic.scalar_utils.integrate_expression.
    backend_kws : None or dict
        keyword arguments passed to symintegral.integrate. Only used for
        "symint".
        
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
                                           "dummy_vars": dummy_vars,
                                           "backend": backend,
                                           "backend_kws": backend_kws},
                              parallel = parallel,
                              symmetry = symmetry)

def integrate_separated(M: VectorFunction | MatrixFunction,
                        domain: Reference,
                        variables: AxisVariablesNotSingle,
                        dummy_vars: AxisVariablesNotSingle,
                        parallel: None | bool = None,
                        parallel_threshold : int = 16,
                        symmetry : None | bool = None,
                        backend: str = "sympy",
                        backend_kws: None | dict = None,
                        simplify: bool = False,
                        return_factor: bool = False
                        ) -> VectorFunction | MatrixFunction | tuple:
    """
    Compute the integral of the VectorFunction or MatrixFunction element-wise
    by integrating each distinct term only once. All entries are separated
    into coefficients and terms depending on the variables

        M_ij = sum_k c_ijk * m_k(variables),

    the distinct terms m_k of all entries are collected, integrated (in
    parallel), and the entries are reassembled by back-substitution

        int M_ij = sum_k c_ijk * int m_k.

    Entries of FEM matrices share most of their terms, so far fewer
    integrals are needed than with integrate.

    Parameters
    ----------
    M : symfem.functions.VectorFunction or symfem.functions.MatrixFunction.
        vector or matrix to be integrated.
    domain : symfem.references.Reference
        domain of the integral
    variables : symfem.symbols.AxisVariablesNotSingle
        variables to integrate with respect to.
    dummy_vars :  symfem.symbols.AxisVariablesNotSingle
        dummy variables to use inside the integral.
    parallel : None or bool
        if True, separation, integration of the distinct terms and reassembly
        are parallelized via Pool. If None, automatically switch to
        parallelization if number of entries is larger than
        parallel_threshold.
    parallel_threshold : int
        if parallel is None and number of entries > parallel_threshold,
        parallelization is used.
    symmetry : None or bool
        if True, assume M to be symmetric and only separate the upper
        triangle. Only applies to square matrices.
    backend : str
        integrator for the distinct terms, either "sympy" or "symint". See
        topoptlab.symbolic.scalar_utils.integrate_expression.
    backend_kws : None or dict
        keyword arguments passed to symintegral.integrate. Only used for
        "symint".
    simplify : bool
        if True, simplify the result cheaply: each distinct integral is
        simplified once, the terms of each entry are grouped by equal
        integral values and only the grouped coefficients are expanded and
        factored (sympy.factor_terms). Much faster than simplify_matrix on
        the result, but the expressions are typically 15-20 % longer as no
        full simplification of the entries is done.
    return_factor : bool
        the common prefactor of all entries that does not depend on the
        variables (e. g. the determinant of the jacobian of affine cells or
        material constants) is always pulled out before separation and
        multiplied back at the end. If True, it is not multiplied back but
        returned separately, so that M_integrated*factor is the integral.

    Returns
    -------
    M_integrated : symfem.functions.VectorFunction or symfem.functions.MatrixFunction
        element-wise integrated vector/matrix.
    factor : sympy.Expr
        common prefactor. Only returned if return_factor is True.

    """
    #
    ismatrixfunc = isinstance(M, MatrixFunction)
    if ismatrixfunc:
        nrow, ncol = M.shape[0], M.shape[1]
        if symmetry and nrow != ncol:
            warn("symmetry only applies to square matrices. Symmetry will be ignored.")
            symmetry = False
        inds = [(i, j) for i in range(nrow)
                for j in range(i if symmetry else 0, ncol)]
        entries = [M[i][j] for i, j in inds]
    elif isinstance(M, VectorFunction):
        nrow = M.shape[0]
        inds = [(i,) for i in range(nrow)]
        entries = [M[i] for i in range(nrow)]
    else:
        raise TypeError("M must be either symfem.functions.MatrixFunction ",
                        "or symfem.functions.VectorFunction. Current type: ",
                        type(M))
    # separate entries into coefficients and terms
    variables = list(variables)
    if backend_kws is None:
        backend_kws = {}
    if parallel is None:
        parallel = len(entries) > parallel_threshold
    pool = Pool(cpu_count()) if parallel else None
    try:
        separated = _map(pool, _separate_entry,
                         [(e, variables) for e in entries],
                         "Separation of entries.")
        # extract common prefactor of all entries
        factors = [sep[0] for sep in separated]
        common = common_factor(factors)
        separated = [{m: c*(f/common) for m, c in sep.items()} \
                     for f, (_, sep) in zip(factors, separated)]
        # distinct terms over all entries
        unique = list(dict.fromkeys(m for sep in separated for m in sep))
        # integrate distinct terms
        values = _map(pool, _integrate_term,
                      [(m, domain, variables, list(dummy_vars), backend,
                        backend_kws, simplify) for m in unique],
                      "Integration of distinct terms.")
        values = dict(zip(unique, values))
        # reassemble by back-substitution
        # multiply common prefactor back in during reassembly
        integrated = _map(pool, _reassemble_entry,
                          [({m: values[m] for m in sep}, sep, simplify,
                            Integer(1) if return_factor else common) \
                           for sep in separated],
                          "Reassembly of entries.")
    except BaseException:
        # kill workers immediately on errors/interrupts instead of waiting
        if pool is not None:
            pool.terminate()
            pool = None
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if not ismatrixfunc:
        M_new = VectorFunction(integrated)
    else:
        M_new = [[0 for j in range(ncol)] for i in range(nrow)]
        for (i, j), v in zip(inds, integrated):
            M_new[i][j] = v
            if symmetry:
                M_new[j][i] = v
        M_new = MatrixFunction(M_new)
    if return_factor:
        return M_new, common
    return M_new

def _integrate_term(term: Expr,
                    domain: Reference,
                    variables: list,
                    dummy_vars: list,
                    backend: str,
                    backend_kws: dict,
                    simplify: bool = False) -> Expr:
    """
    Helper function for integrate_separated() that integrates one distinct
    term over the domain.

    Parameters
    ----------
    term : sympy.Expr
        term to be integrated.
    domain : symfem.references.Reference
        domain of the integral
    variables : list
        variables to integrate with respect to.
    dummy_vars : list
        dummy variables to use inside the integral.
    backend : str
        either "sympy" or "symint".
    backend_kws : dict
        keyword arguments passed to the integrator of the backend.
    simplify : bool
        if True, simplify the integral.

    Returns
    -------
    integrated : sympy.Expr
        integral of term.

    """
    integrated = integrate_expression(expression=term, domain=domain,
                                      variables=variables,
                                      dummy_vars=dummy_vars,
                                      backend=backend,
                                      backend_kws=backend_kws)
    if simplify:
        return sympify(integrated).simplify()
    return integrated

def _separate_entry(entry: ScalarFunction,
                    variables: list) -> dict:
    """
    Helper function for integrate_separated() that separates one entry into
    terms depending on the variables and their coefficients.

    Parameters
    ----------
    entry : symfem.functions.ScalarFunction
        entry of matrix/vector.
    variables : list
        variables to integrate with respect to.

    Returns
    -------
    factor : sympy.Expr
        factor of the entry that does not depend on the variables.
    separated : dict
        keys are the terms, values the coefficients of the remainder.

    """
    factor, remainder = _factor_expr(expression=entry.as_sympy(),
                                     variables=variables)
    return factor, separate_terms(expression=remainder, variables=variables)

def _reassemble_entry(values: dict,
                      separated: dict,
                      simplify: bool,
                      factor: Expr = Integer(1)) -> Expr:
    """
    Helper function for integrate_separated() that reassembles one entry
    from the integrated terms.

    Parameters
    ----------
    values : dict
        integrals of the terms of the entry.
    separated : dict
        keys are the terms, values the coefficients.
    simplify : bool
        if True, group the terms by equal integral values and expand and
        factor the grouped coefficients.
    factor : sympy.Expr
        prefactor multiplied into the coefficients before expansion, so
        that it can cancel with the coefficients.

    Returns
    -------
    integrated : sympy.Expr
        integrated entry.

    """
    if not simplify:
        return factor*sum((c*values[m] for m, c in separated.items()),
                          Integer(0))
    grouped = {}
    for m, c in separated.items():
        if values[m] != 0:
            grouped[values[m]] = grouped.get(values[m], 0) + c
    return factor_terms(sum((factor_terms(expand(factor*c))*v \
                             for v, c in grouped.items()), Integer(0)))

def _map(pool: None | Pool,
         func: Callable,
         tasks: list,
         dscr: str) -> list:
    """
    Apply func to the tasks serially or with the given Pool.

    Parameters
    ----------
    pool : None or multiprocessing.Pool
        pool. If None, serial execution.
    func : callable
        function applied to each task.
    tasks : list
        list of argument tuples.
    dscr : str
        description of function displayed on the progress bar.

    Returns
    -------
    results : list
        results in the order of the tasks.

    """
    if pool is None:
        return [func(*task) for task in tasks]
    # imap (unlike starmap) yields results as they finish, so the progress
    # bar shows the actual progress
    return list(tqdm(pool.imap(_star_apply, [(func, task) for task in tasks]),
                     total=len(tasks), desc=dscr))

def _star_apply(func_task: tuple) -> Any:
    """
    Unpack (func, task) and call func(*task). Helper for Pool.imap which only
    passes a single argument.

    Parameters
    ----------
    func_task : tuple
        function and tuple of its arguments.

    Returns
    -------
    result : Any
        return value of func.

    """
    func, task = func_task
    return func(*task)

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

def diag(v: list | MatrixFunction) -> list | MatrixFunction:
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
          mode: str="symfem") -> ScalarFunction:
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
            Adet: None | ScalarFunction = None) -> MatrixFunction:
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
        s: float | int) -> MatrixFunction:
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

def check_square(M: MatrixFunction) -> None:
    """
    Raise a ValueError if MatrixFunction M is not a square matrix.

    Parameters
    ----------
    M : symfem.functions.MatrixFunction
        matrix function.

    Returns
    -------
    None
    """
    if not is_square(M):
        raise ValueError("M must be square. Current shape: ", M.shape)

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