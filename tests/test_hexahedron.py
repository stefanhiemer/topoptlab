from numpy import array,arange,stack,vstack
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.trilinear_hexahedron import shape_functions

@pytest.mark.parametrize('xi, eta, zeta, outcome',
                         [(-1,-1,-1,array([[1.,0,0,0,0,0,0,0]])),
                         (1,-1,-1,array([[0,1.,0,0,0,0,0,0]])),
                         (1,1,-1,array([[0,0,1.,0,0,0,0,0]])),
                         (-1,1,-1,array([[0,0,0,1.,0,0,0,0]])),
                         (-1,-1,1,array([[0,0,0,0,1.,0,0,0]])),
                         (1,-1,1,array([[0,0,0,0,0,1.,0,0]])),
                         (1,1,1,array([[0,0,0,0,0,0,1.,0]])),
                         (-1,1,1,array([[0,0,0,0,0,0,0,1.]])),
                         (array([-1,1,1,-1,-1,1,1,-1]),
                          array([-1,-1,1,1,-1,-1,1,1]),
                          array([-1,-1,-1,-1,1,1,1,1]),
                          array([[1.,0,0,0,0,0,0,0],
                                 [0,1.,0,0,0,0,0,0],
                                 [0,0,1.,0,0,0,0,0],
                                 [0,0,0,1.,0,0,0,0],
                                 [0,0,0,0,1.,0,0,0],
                                 [0,0,0,0,0,1.,0,0],
                                 [0,0,0,0,0,0,1.,0],
                                 [0,0,0,0,0,0,0,1.]]))])

def test_shapefunction(xi,eta,zeta,outcome):
    assert_almost_equal(shape_functions(xi,eta,zeta),outcome) 
    return

from topoptlab.elements.trilinear_hexahedron import bmatrix, bmatrix_cuboid

@pytest.mark.parametrize('xe, xi, eta, zeta, a, b, c, all_elems',
                         [(2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           0,0,0,[4],[4],[4],False),
                          (2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           1,1,1,[4],[4],[4],False),
                          (2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                            array([0]),array([0]),array([0]),
                            [4],[4],[4],
                            False),
                          (2*array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                     [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]),
                           array([0,1]),array([0,1]),array([0,1]),
                           [4],[4],[4],
                           False),
                          (array([[[-2,-2,-2],[2,-2,-2],[2,2,-2],[-2,2,-2],
                                   [-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],
                                  [[-3,-3,-3],[3,-3,-3],[3,3,-3],[-3,3,-3],
                                   [-3,-3,3],[3,-3,3],[3,3,3],[-3,3,3]]]),
                            array([0]),array([0]),array([0]),
                            [4,6],[4,6],[4,6],
                            False),
                          (array([[[-2,-2,-2],[2,-2,-2],[2,2,-2],[-2,2,-2],
                                   [-2,-2,2],[2,-2,2],[2,2,2],[-2,2,2]],
                                  [[-3,-3,-3],[3,-3,-3],[3,3,-3],[-3,3,-3],
                                   [-3,-3,3],[3,-3,3],[3,3,3],[-3,3,3]]]),
                            arange(7)/6,arange(-6,1)/7,arange(-6,1)/7,
                            [4,6],[4,6],[4,6],
                            True),])

def test_bmatrix(xe,xi,eta,zeta,a,b,c,all_elems):
    B = vstack([bmatrix_cuboid(xi,eta,zeta,_a,_b,_c) for _a,_b,_c in zip(a,b,c)])
    assert_almost_equal(bmatrix(xi,eta,zeta,xe,all_elems),B) 
    return
