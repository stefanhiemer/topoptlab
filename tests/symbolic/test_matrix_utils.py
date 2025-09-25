from symfem.functions import MatrixFunction

from topoptlab.symbolic.matrix_utils import eye,diag,to_square

import pytest


@pytest.mark.parametrize('ndim,sol',
                         [(3,
                           MatrixFunction([[1, 0, 0],
                                           [0, 1, 0],
                                           [0, 0, 1]])) ])

def test_eye(ndim,sol):
    
    assert sol == eye(ndim)
    return

@pytest.mark.parametrize('diagonal,sol',
                         [([1,2,3],
                           MatrixFunction([[1, 0, 0],
                                           [0, 2, 0],
                                           [0, 0, 3]]))  ])

def test_diag(diagonal,sol):
    
    assert sol == diag(diagonal)
    return

@pytest.mark.parametrize('size,order,sol',
                         [(9,"F",
                           MatrixFunction([[0, 3, 6],
                                           [1, 4, 7],
                                           [2, 5, 8]])),
                           (9,"C",
                             MatrixFunction([[0, 1, 2],
                                             [3, 4, 5],
                                             [6, 7, 8]])) ])

def test_tosquare(size,order,sol):
    v = MatrixFunction([[i] for i in range(size)])
    assert sol == to_square(v,order=order)
    return

