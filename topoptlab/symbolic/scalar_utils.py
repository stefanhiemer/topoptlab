# SPDX-License-Identifier: GPL-3.0-or-later
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sympy import Expr, sympify, Integral
from symintegral import integrate as symint_integrate
from symfem.references import Reference
from symfem.symbols import AxisVariablesNotSingle
from symfem.functions import ScalarFunction,VectorFunction

from topoptlab.symbolic.utils import split_expression

def integral(scalar_function: ScalarFunction,
             domain: Reference,
             vars: AxisVariablesNotSingle,
             dummy_vars: AxisVariablesNotSingle,
             parallel: bool = False,
             backend: str = "sympy",
             backend_kws: None | dict = None) -> ScalarFunction:
    """
    Compute the integral of a scalar function. If parallel, the expression
    is split into chunks of common powers of the variables that are
    integrated in parallel.

    Parameters
    ----------
    scalar_function : symfem.functions.ScalarFunction
        function to be integrated.
    domain : symfem.references.Reference
        domain of the integral.
    vars : symfem.symbols.AxisVariablesNotSingle
        variables to integrate with respect to.
    dummy_vars : symfem.symbols.AxisVariablesNotSingle
        dummy variables to use inside the integral.
    parallel : bool
        if True, integrate chunks of the expression in parallel.
    backend : str
        integrator, either "sympy" or "symint". See integrate_expression.
    backend_kws : None or dict
        keyword arguments passed to symintegral.integrate. Only used for
        "symint".

    Returns
    -------
    integrated : symfem.functions.ScalarFunction
        integral.
    """
    out = scalar_function._f * 1
    if not parallel:
        return ScalarFunction(integrate_expression(expression=out,
                                                   domain=domain,
                                                   variables=vars,
                                                   dummy_vars=dummy_vars,
                                                   backend=backend,
                                                   backend_kws=backend_kws))
    else:
        chunks = split_expression(expression=out,
                                variables=vars,
                                include_nonlin=False,
                                nchunks=cpu_count())
        args = [(c, domain, vars, dummy_vars, backend, backend_kws)
                for c in chunks]
        with Pool(cpu_count()) as pool:
            chunks = list(tqdm(pool.starmap(integrate_expression, args),
                            total=len(args),
                            desc="Parallel integration of chunks"))
        return ScalarFunction(sum(chunks))

def integrate_expression(expression: Expr,
                         domain: Reference,
                         variables: AxisVariablesNotSingle,
                         dummy_vars: AxisVariablesNotSingle,
                         backend: str = "sympy",
                         backend_kws: None | dict = None) -> Expr:
    """
    Integrate an expression over the domain. The variables are mapped to the
    reference cell via the dummy variables as in symfem by

       x = origin + sum([t_i*a_i for t_i, a_i in zip(t, axes)])

    where a_i is the i-th axis of the domain and t_i the corresponding dummy
    variable on symfem's unit cell [0,1].

    Parameters
    ----------
    expression : sympy.Expr or symfem.functions.ScalarFunction
        expression to be integrated.
    domain : symfem.references.Reference
        domain of the integral.
    variables : symfem.symbols.AxisVariablesNotSingle
        variables to integrate with respect to.
    dummy_vars : symfem.symbols.AxisVariablesNotSingle
        dummy variables to use inside the integral.
    backend : str
        integrator. "sympy" uses sympy.integrate, "symint" uses
        symintegral.integrate (closed-form rules, then the CAS back ends of
        symintegral), e. g. for rational integrands of non-affine elements.
    backend_kws : None or dict
        keyword arguments passed to symintegral.integrate. Only used for
        "symint".

    Returns
    -------
    integrated : sympy.Expr
        integral of expression.

    """
    # check if ScalarFunction
    if hasattr(expression, "as_sympy"):
        expression = expression.as_sympy()
    # write coordinates in symfem's dummy coordinate variables t
    point = VectorFunction(domain.origin)
    for ti, a in zip(dummy_vars, domain.axes):
        point += ti * VectorFunction(a)
    point = [p.as_sympy() for p in point]
    limits = domain.integration_limits(dummy_vars)
    if backend == "sympy":
        return _integrate_chunk(expr=expression,
                                vars=variables,
                                dummy_vars=dummy_vars,
                                orig=point,
                                jacobian=domain.jacobian(),
                                limits=limits)
    elif backend == "symint":
        expr = sympify(expression)
        for v, p in zip(variables, point):
            expr = expr.subs(v, p)
        return symint_integrate(Integral(expr*domain.jacobian(),
                                         *limits),
                                **(backend_kws or {}))
    else:
        raise ValueError(f"unknown backend: {backend}")

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
    expr = sympify(expr)
    #
    for v, p in zip(vars, orig):
        expr = expr.subs(v, p)
    if len(limits[0]) == 2:
        for i in limits:
            assert len(i) == 2
            expr = expr.subs(*i)
        return expr
    else:
        expr *= jacobian
        #
        return expr.integrate(*limits)