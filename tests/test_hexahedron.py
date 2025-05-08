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

from topoptlab.elements.trilinear_hexahedron import create_edofMat, apply_pbc

@pytest.mark.parametrize('pbc, nnode_dof, target',
                         [((False,False,True), 1, 
                           array([[ 1,  4,  3,  0, 10, 13, 12,  9],
                                  [ 2,  5,  4,  1, 11, 14, 13, 10],
                                  [ 4,  7,  6,  3, 13, 16, 15, 12],
                                  [ 5,  8,  7,  4, 14, 17, 16, 13],
                                  [10, 13, 12,  9,  1,  4,  3,  0],
                                  [11, 14, 13, 10,  2,  5,  4,  1],
                                  [13, 16, 15, 12,  4,  7,  6,  3],
                                  [14, 17, 16, 13,  5,  8,  7,  4]])),
                          ((False,True,False), 1, 
                            array([[ 1,  3,  2,  0,  7,  9,  8,  6],
                                   [ 0,  2,  3,  1,  6,  8,  9,  7],
                                   [ 3,  5,  4,  2,  9, 11, 10,  8],
                                   [ 2,  4,  5,  3,  8, 10, 11,  9],
                                   [ 7,  9,  8,  6, 13, 15, 14, 12],
                                   [ 6,  8,  9,  7, 12, 14, 15, 13],
                                   [ 9, 11, 10,  8, 15, 17, 16, 14],
                                   [ 8, 10, 11,  9, 14, 16, 17, 15]])),
                          ((True,False,False), 1, 
                            array([[1, 4, 3, 0, 7, 10, 9, 6],
                                   [2, 5, 4, 1, 8, 11, 10, 7],
                                   [4, 1, 0, 3, 10, 7, 6, 9],
                                   [5, 2, 1, 4, 11, 8, 7, 10],
                                   [7, 10, 9, 6, 13, 16, 15, 12],
                                   [8, 11, 10, 7, 14, 17, 16, 13],
                                   [10, 7, 6, 9, 16, 13, 12, 15],
                                   [11, 8, 7, 10, 17, 14, 13, 16]])),
                          ((True,True,False), 1, 
                            array([[1, 3, 2, 0, 5, 7, 6, 4],
                                   [0, 2, 3, 1, 4, 6, 7, 5],
                                   [3, 1, 0, 2, 7, 5, 4, 6],
                                   [2, 0, 1, 3, 6, 4, 5, 7],
                                   [5, 7, 6, 4, 9, 11, 10, 8],
                                   [4, 6, 7, 5, 8, 10, 11, 9],
                                   [7, 5, 4, 6, 11, 9, 8, 10],
                                   [6, 4, 5, 7, 10, 8, 9, 11]]))])

def test_pbc(pbc,nnode_dof,target):
    nelx,nely,nelz=2,2,2
    
    edofMat = create_edofMat(nelx=nelx, nely=nely, nelz=nelz, 
                             nnode_dof=nnode_dof)[0]
    assert_almost_equal(apply_pbc(edofMat=edofMat,pbc=pbc,
                                  nelx=nelx,nely=nely,nelz=nelz,
                                  nnode_dof=nnode_dof),
                        target)
    return
