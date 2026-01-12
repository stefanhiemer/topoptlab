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

from topoptlab.symbolic.matrix_utils import to_voigt,generate_constMatrix

@pytest.mark.parametrize('ndim',
                         [ (2),
                           (3)])

def test_tovoigt(ndim):
    #
    M = generate_constMatrix(ndim,ndim,"M",symmetric=True)
    M_v = to_voigt(M)
    if ndim==2:
        sol = MatrixFunction( [[M[0,0]],[M[1,1]],
                               [M[0,1]]] )
    elif ndim==3:
        sol = MatrixFunction( [[M[0,0]],[M[1,1]],[M[2,2]],
                               [M[1,2]],[M[0,2]],[M[0,1]]] )
    assert sol == M_v
    return

