# SPDX-License-Identifier: GPL-3.0-or-later
from typing import List,Union
from multiprocessing import Pool, cpu_count
from functools import partial 

from tqdm import tqdm

from sympy import Expr,Symbol
from symfem.references import Reference
from symfem.symbols import AxisVariablesNotSingle
from symfem.functions import Function,ScalarFunction,VectorFunction

from topoptlab.symbolic.utils import split_expression

def integral(scalar_function: ScalarFunction, 
             domain: Reference,
             vars: AxisVariablesNotSingle,
             dummy_vars: AxisVariablesNotSingle, 
             parallel: bool = False) -> ScalarFunction:
    """Compute the integral of the function.

    Args:
        domain: The domain of the integral
        vars: The variables to integrate with respect to
        dummy_vars: The dummy variables to use inside the integral

    Returns:
        The integral
    """
    limits = domain.integration_limits(dummy_vars)
    point = VectorFunction(domain.origin)
    for ti, a in zip(dummy_vars, domain.axes):
        point += ti * VectorFunction(a)
    out = scalar_function._f * 1
    #
    jacobian = domain.jacobian()
    #
    if parallel:
        print("Parallel")
        #
        chunks = split_expression(expression=out,
                                  variables=vars,
                                  include_nonlin=False,
                                  nchunks=None)
        #
        args = [(c, vars, dummy_vars, point, jacobian, limits)
                for c in chunks]
        #
        with Pool(cpu_count()) as pool:
            chunks = list(tqdm(pool.starmap(_integrate_chunk, args),
                                  total=len(args),
                                  desc="Parallel integration of chunks"))
        final = 0 
        for i in range(len(chunks)):
            final += chunks[i] 
        return ScalarFunction(final)
    else:
        return ScalarFunction(_integrate_chunk(expr=out,
                                               vars=vars,
                                               dummy_vars=dummy_vars,
                                               orig=point,
                                               jacobian=jacobian,
                                               limits=limits))

def _integrate_chunk(expr : Expr, 
                     vars : AxisVariablesNotSingle,
                     dummy_vars : AxisVariablesNotSingle, 
                     orig : int,
                     jacobian: Expr,
                     limits: Expr,
                     ) -> Expr:
    """Integrate sympy expression over domain b substituting vars with 
    dummy_vars.

    Args:
        limits: The variables and limits

    Returns:
        The integral
    """
    #
    for v, p in zip(vars, orig):
        expr = expr.subs(v, p)
    if len(limits[0]) == 2:
        for i in limits:
            assert len(i) == 2
            expr = expr.subs(*i)
        return expr

    expr *= jacobian
    #
    return expr.integrate(*limits)