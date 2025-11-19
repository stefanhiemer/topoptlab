from numpy import arange,array
from numpy.testing import assert_equal


from topoptlab.blocksparse.make_blocks import create_equal_blocks

import pytest

@pytest.mark.parametrize('nelx, nely, nblocks, nnode_dof, solution',
                         [(4,4,4,1,
                           [array([0, 1, 2, 3, 4, 5]), 
                            array([6, 7, 8, 9, 10, 11]), 
                            array([12, 13, 14, 15, 16, 17]), 
                            array([18, 19, 20, 21, 22, 23, 24])]),
                          (4,4,4,2,
                           [array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                                   11]), 
                            array([12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 
                                   23]), 
                            array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                                   35]), 
                            array([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                                   47, 48, 49])])])

def test_create_equal(nelx,nely,nblocks,nnode_dof, solution):
    
    blocks = create_equal_blocks(n_nodes=int((nelx+1)*(nely+1)) , 
                                 nblocks=nblocks,
                                 nnode_dof=nnode_dof)
    #
    assert_equal(solution, 
                 blocks)
    return

from topoptlab.elements.bilinear_quadrilateral import create_edofMat
from topoptlab.blocksparse.make_blocks import create_volthresh_blocks

@pytest.mark.parametrize('nelx, nely, volfrac, nnode_dof, solution',
                         [(4,4,0.5,1,
                           [array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                                   21, 22, 23, 24]), 
                            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]),
                          (4,4,0.5,2,
                           [array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                                   31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 
                                   42, 43, 44, 45, 46, 47, 48, 49]), 
                            array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                                   11, 12, 13, 14, 15, 16, 17, 18, 19])])])

def test_create_volthresh(nelx,nely,volfrac,nnode_dof, solution):
    
    #
    xPhys = (arange(1,nelx*nely+1)/nelx*nely)[:,None]
    #
    blocks = create_volthresh_blocks(xPhys=xPhys,
                            volfrac=volfrac,
                            edofMat=create_edofMat(nelx=nelx,
                                                   nely=nely,
                                                   nelz=None,
                                                   nnode_dof=nnode_dof)[0])
    #
    assert_equal(solution, 
                 blocks)
    return

from topoptlab.blocksparse.make_blocks import create_quantile_blocks

@pytest.mark.parametrize('nelx, nely, quantiles, nnode_dof, solution',
                         [(4,4,[0.25,0.5,0.75],1,
                           [array([15, 16, 17, 18, 19, 20, 21, 22, 23, 24]), 
                            array([10, 11, 12, 13, 14]), 
                            array([5, 6, 7, 8, 9]), 
                            array([0, 1, 2, 3, 4])]),
                          (4,4,[0.25,0.5,0.75],2,
                           [array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
                                   41, 42, 43, 44, 45, 46, 47, 48, 49]), 
                            array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29]), 
                            array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]), 
                            array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])])])

def test_create_quantile(nelx,nely,quantiles,nnode_dof, solution):
    
    #
    xPhys = (arange(1,nelx*nely+1)/nelx*nely)[:,None]
    #
    blocks = create_quantile_blocks(xPhys=xPhys,
                            quantiles=quantiles,
                            edofMat=create_edofMat(nelx=nelx,
                                                   nely=nely,
                                                   nelz=None,
                                                   nnode_dof=nnode_dof)[0])
    #
    assert_equal(solution, 
                 blocks)
    return
