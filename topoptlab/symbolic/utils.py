# SPDX-License-Identifier: GPL-3.0-or-later
from sympy import expand,simplify,Expr

def is_equal(expr1: Expr, expr2: Expr)-> bool:
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
