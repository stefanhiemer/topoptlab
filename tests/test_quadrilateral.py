from numpy import array,arange,stack,vstack
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.bilinear_quadrilateral import shape_functions

@pytest.mark.parametrize('xi, eta, outcome',
                         [(-1,-1,array([[1.,0,0,0]])),
                         (1,-1,array([[0,1.,0,0]])),
                         (1,1,array([[0,0,1.,0]])),
                         (-1,1,array([[0,0,0,1.]])),
                         (array([-1,1,1,-1]),
                          array([-1,-1,1,1]),
                          array([[1.,0,0,0],
                                 [0,1.,0,0],
                                 [0,0,1.,0],
                                 [0,0,0,1.]]))])

def test_shapefunction(xi,eta,outcome):
    assert_almost_equal(shape_functions(xi,eta),outcome) 
    return

from topoptlab.elements.bilinear_quadrilateral import jacobian, jacobian_rectangle

@pytest.mark.parametrize('xe, xi, eta, a, b, all_elems',
                         [(2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           0,0,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1,1,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0]),array([0]),[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0,1]),array([0,1]),[4,4],[4,4],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            array([0]),array([0]),[4,6],[4,6],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            arange(7)/6,arange(-6,1)/7,[4,6],[4,6],True),])

def test_jacobian(xe,xi,eta,a,b,all_elems):
    
    if not all_elems:
        j = stack([jacobian_rectangle(_a,_b) for _a,_b in zip(a,b)])
    else:
        j = []
        for i in range(len(a)):
            j = j + [jacobian_rectangle(a[i],b[i]) for _ in range(xi.shape[0])]
        j = stack(j)
    assert_almost_equal(jacobian(xi,eta,xe,all_elems),j) 
    return

from topoptlab.elements.bilinear_quadrilateral import invjacobian, invjacobian_rectangle

@pytest.mark.parametrize('xe, xi, eta, a, b, all_elems',
                         [(2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           0,0,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1,1,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0]),array([0]),[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0,1]),array([0,1]),[4,4],[4,4],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            array([0]),array([0]),[4,6],[4,6],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            arange(7)/6,arange(-6,1)/7,[4,6],[4,6],True),])

def test_invjacobian(xe,xi,eta,a,b,all_elems):
    if not all_elems:
        jinv = stack([invjacobian_rectangle(_a,_b) for _a,_b in zip(a,b)])
    else:
        jinv = []
        for i in range(len(a)):
            jinv = jinv + [invjacobian_rectangle(a[i],b[i]) for _ in range(xi.shape[0])]
        jinv = stack(jinv)
    assert_almost_equal(invjacobian(xi,eta,xe,all_elems),jinv) 
    return

from topoptlab.elements.bilinear_quadrilateral import bmatrix,bmatrix_rectangle

@pytest.mark.parametrize('xe, xi, eta, a, b, all_elems',
                         [(2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           0,0,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           1,1,[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0]),array([0]),[4],[4],False),
                          (2*array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                            array([0,1]),array([0,1]),[4],[4],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            array([0]),array([0]),[4,6],[4,6],False),
                          (array([[[-2,-2],[2,-2],[2,2],[-2,2]],
                                  [[-3,-3],[3,-3],[3,3],[-3,3]]]),
                            arange(7)/6,arange(-6,1)/7,[4,6],[4,6],True),])

def test_bmatrix(xe,xi,eta,a,b,all_elems):
    B = vstack([bmatrix_rectangle(xi,eta,_a,_b) for _a,_b in zip(a,b)])
    assert_almost_equal(bmatrix(xi,eta,xe,all_elems),B) 
    return
