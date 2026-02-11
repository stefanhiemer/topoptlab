from sympy import symbols, exp,sin

from topoptlab.symbolic.utils import is_equal, split_expression



import pytest


@pytest.mark.parametrize('nchunks',
                         [2])

def test_consistency_split_expression(nchunks):
    
    # symbols
    a, x, y = symbols('a x y')
    
    # test expression
    expr = (a*2*x**2*exp(x**2)
            + 3*x**2*exp(x)*sin(a)
            + 4*y*sin(x)*exp(a)
            + 5*exp(x)
            + 6)
    
    # split
    chunks = split_expression(expression=expr,
                              variables=[x, y],
                              include_nonlin=True,
                              nchunks=nchunks)
    
    # check reconstruction
    assert is_equal(sum(chunks), expr)
    return
