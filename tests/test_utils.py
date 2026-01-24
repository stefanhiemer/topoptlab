from numpy import arange,array,column_stack,prod
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

@pytest.mark.parametrize('nelx, nely',
                         [(3,2),
                          (4,2),
                          (9,5)])

def test_mapimg(nelx,nely):
    x = arange(nelx*nely)
    img = map_eltoimg(quant=x,
                nelx=nelx,
                nely=nely)
    
    assert_almost_equal(x,
                        map_imgtoel(img=img,
                                    nelx=nelx,
                                    nely=nely)) 
    return

@pytest.mark.parametrize('nelx, nely, nelz',
                         [(3,2,4),
                          (4,2,3),
                          (9,5,7)])

def test_mapvoxel(nelx,nely,nelz):
    x = arange(nelx*nely*nelz)
    voxel = map_eltovoxel(quant=x,
                        nelx=nelx,
                        nely=nely,
                        nelz=nelz)
    assert_almost_equal(x,
                        map_voxeltoel(voxel=voxel,
                                      nelx=nelx,
                                      nely=nely,
                                      nelz=nelz)) 
    return

from topoptlab.utils import elid_to_coords

@pytest.mark.parametrize('nelx, nely, nelz, solution',
                         [(2,3,None,
                           (array([0, 0, 0, 1, 1, 1]), 
                            array([2, 1, 0, 2, 1, 0]))),
                          (2,3,2,
                           (array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]), 
                            array([2, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0]), 
                            array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])))])

def test_elidtocoords(nelx, nely, nelz, solution):
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    elid = arange( prod( [nelx,nely,nelz][:ndim]).astype(int) )
    #
    assert_almost_equal(elid_to_coords(el=elid,
                                       nelx=nelx,nely=nely,nelz=nelz),
                        column_stack(solution)) 
    
    return

from topoptlab.utils import nodeid_to_coords

@pytest.mark.parametrize('nelx, nely, nelz, solution',
                         [(2,3,None,
                           (array([-0.5, -0.5, -0.5,-0.5,
                                    0.5, 0.5, 0.5, 0.5, 
                                    1.5, 1.5, 1.5, 1.5]), 
                            array([2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5,1.5,0.5,-0.5]))),
                          (2,3,2,
                           (array([-0.5, -0.5, -0.5, -0.5, 
                                   0.5, 0.5, 0.5, 0.5,
                                   1.5, 1.5, 1.5, 1.5,
                                   -0.5, -0.5, -0.5, -0.5, 
                                   0.5, 0.5, 0.5, 0.5,
                                   1.5, 1.5, 1.5, 1.5,
                                   -0.5, -0.5, -0.5, -0.5, 
                                   0.5, 0.5, 0.5, 0.5,
                                   1.5, 1.5, 1.5, 1.5]), 
                            array([2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5,
                                   2.5, 1.5, 0.5, -0.5]), 
                            array([-0.5, -0.5, -0.5, -0.5, 
                                   -0.5, -0.5, -0.5, -0.5,
                                   -0.5, -0.5, -0.5, -0.5, 
                                   0.5, 0.5, 0.5, 0.5, 
                                   0.5, 0.5, 0.5, 0.5, 
                                   0.5, 0.5, 0.5, 0.5, 
                                   1.5, 1.5, 1.5, 1.5, 
                                   1.5, 1.5, 1.5, 1.5,
                                   1.5, 1.5, 1.5, 1.5])))])

def test_ndidtocoords(nelx, nely, nelz, solution):
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    ndid = arange( prod( array([nelx,nely,nelz][:ndim])+1).astype(int) )
    #
    assert_almost_equal(nodeid_to_coords(nd=ndid,
                                         nelx=nelx,nely=nely,nelz=nelz),
                        column_stack(solution)) 
    
    return

from topoptlab.utils import upsampling

@pytest.mark.parametrize('nelx, nely, nelz, magn, solution',
                         [(2,2,None,2,
                           array([[0],[0],[1],[1],
                                  [0],[0],[1],[1],
                                  [2],[2],[3],[3],
                                  [2],[2],[3],[3]])),])

def test_upsampling(nelx,nely,nelz,magn,solution):
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    x = arange( prod( [nelx,nely,nelz][:ndim]).astype(int) )[:,None]
    #
    assert_almost_equal(solution,
                        upsampling(x=x,nelx=nelx,nely=nely,nelz=nelz,
                                   magnification=magn)) 
    return


from topoptlab.utils import voigt_index, voigt_pair  


@pytest.mark.parametrize("ndim", [2,3])
def test_voigt_inverse_consistency(ndim):
    for i in range(ndim):
        for j in range(ndim):
            alpha = voigt_index(i, j, ndim)
            ii, jj = voigt_pair(alpha, ndim)

            ii = int(ii)
            jj = int(jj)

            assert set((i, j)) == set((ii, jj))
    return
