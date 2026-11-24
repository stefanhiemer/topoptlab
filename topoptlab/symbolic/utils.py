# SPDX-License-Identifier: GPL-3.0-or-later
from multiprocessing import cpu_count

import numpy as np

from sympy import expand, simplify, Expr, collect, Function, Piecewise,\
                  Add, sympify

def is_equal(expr1: Expr,
             expr2: Expr)-> bool:
    """
    Return True if two SymPy expressions are algebraically identical.

    The check expands the difference and asks SymPy to simplify it to zero.

    Parameters
    ----------
    expr1 : sympy.Expr
        first expression.
    expr2 : sympy.Expr
        second expression.

    Returns
    -------
    bool
        True if ``expr1 - expr2`` simplifies to zero (identical),
        False otherwise.
    """
    return simplify(expand(expr1 - expr2)) == 0

def split_expression(expression: Expr,
                     variables: list,
                     include_nonlin: bool,
                     nchunks: None | int) -> list:
    """
    Split expression into common powers of provided variables and merge to n
    chunks for independent parallel operations on each chunk.
    

    Parameters
    ----------
    expression : Expr
        sympy expression.
    variables : list
        list of sympy variables.
    include_nonlin : bool
        if True, also
    nchunks : Union[None,int]
        number of chunks. If None, number of processors.

    Returns
    -------
    chunked_expression : list
        list of additive sympy expressions whose sum equals `expression`.

    """
    if isinstance(variables, tuple):
        variables = list(variables)
    #
    if nchunks is None:
        nchunks = cpu_count()
    # extract nonlinear functions and treat them as if it were a polynomial 
    # power of a new variable
    if include_nonlin:
        #
        funcs = [f for f in expression.atoms(Function) \
                 if f.free_symbols & set(variables)]
    else:
        funcs = []
    # transform expression to list of terms of common powers
    terms = collect(expression, variables+funcs,
                    evaluate=False)
    terms = [terms[key]*key for key in terms.keys()]
    # determine number of chunks
    nterms = len(terms)
    if nterms < nchunks:
        chunk_size = 1
    else: 
        chunk_size = nterms // nchunks
    #
    chunks = [sum(terms[i*chunk_size:(i+1)*chunk_size]) \
              for i in range(nchunks-1)]
    chunks.append(sum(terms[(nchunks-1)*chunk_size:]))
    return chunks



def separate_terms(expression: Expr,
                   variables: list) -> dict:
    """
    Separate an expression into terms that depend on the provided variables
    and coefficients that do not, i. e.

        expression = sum_k c_k * m_k(variables)

    The expression is expanded first, so terms with a common denominator
    depending on the variables stay together with that denominator. Used to
    integrate a matrix by integrating each distinct term m_k only once.

    Parameters
    ----------
    expression : sympy.Expr
        sympy expression.
    variables : list
        list of sympy variables.

    Returns
    -------
    terms : dict
        keys are the terms m_k that depend on the variables (1 for the
        constant part), values the coefficients c_k.

    """
    terms = {}
    for term in Add.make_args(expand(sympify(expression))):
        coeff, dep = term.as_independent(*variables, as_Add=False)
        terms[dep] = terms.get(dep, 0) + coeff
    return terms

def _choose_branch(piecewise_part) -> Expr:
    """
    Eliminate Piecewise heuristically by taking the branch with the longest 
    condition in terms of its string representation.
    
    Parameters
    ----------
    piecewise_part : Expr
        sympy Piecewise expression.

    Returns
    -------
    expression_branch : Expr
        main branch of given expression.

    """
    # remove trivial condition (condition == True)
    nontrivial = [(expr,cond) for expr,cond in piecewise_part.args \
                  if cond != True]
    # if only trivial exists, return it
    if not nontrivial:
        return piecewise_part.args[-1][0]
    # heuristic criterion: choose branch with longest string representation
    best = max(nontrivial, key=lambda ec: len(str(ec[1])))
    return best[0]

def take_generic_branch(expression):
    """
    Eliminate piecewise definitions (Piecewise) heuristically by taking the 
    branch with the longest condition in terms of its string representation.
    
    Parameters
    ----------
    expression : Expr or symfem.functions.ScalarFunction
        sympy expression with Piecewise.

    Returns
    -------
    expression_branch : Expr
        main branch of given expression.

    """
    if hasattr(expression, "as_sympy"):
        expression = sympify(expression.as_sympy())
    return expression.replace(lambda e: isinstance(e,Piecewise),
                              _choose_branch)