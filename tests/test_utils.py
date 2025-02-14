from numpy import arange
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

@pytest.mark.parametrize('nelx, nely',
                         [(3,2),
                          (4,2),
                          (9,5)])

def test_maptoimg(nelx,nely):
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

def test_maptovoxel(nelx,nely,nelz):
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