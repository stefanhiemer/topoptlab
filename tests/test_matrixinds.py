from numpy import unique,sort,argsort,lexsort
from numpy import hstack,vstack,column_stack,flip,split
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d,lk_linear_elast_2d,lk_linear_elast_aniso_2d
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as edofMat3d
from topoptlab.fem import create_matrixinds
from topoptlab.utils import unique_sort
@pytest.mark.parametrize('nelx, nely, nelz, nnode_dof',
                         [(3,2,None,1),
                          (4,8,None,1),
                          (4,8,None,2),
                          (3,2,1,1),
                          (4,8,2,1),
                          (4,8,2,2),])

def test_inds2d(nelx, nely, nelz, nnode_dof):
    if nelz is None:
        # 
        edofMat,_,_,_,_= edofMat2d(nelx=nelx,nely=nely,
                                   nnode_dof=nnode_dof)
    else:
        # 
        edofMat,_,_,_,_= edofMat3d(nelx=nelx,nely=nely,nelz=nelz,
                                   nnode_dof=nnode_dof)
    # 
    iM,jM = create_matrixinds(edofMat,mode="full")
    M_full = unique_sort(iM, jM,combine=True)
    #
    _iM,_jM = create_matrixinds(edofMat,mode="lower")
    M_lower = column_stack((_iM,_jM))
    # for comparison must duplicate and avoid double mentioning diagonal entries
    mask = M_lower[:,0] != M_lower[:,1]
    M_lower = vstack((M_lower,
                      flip(M_lower[mask],axis=1)))
    M_lower = unique_sort(M_lower[:,0], M_lower[:,1], 
                          combine=True)
    assert_almost_equal(M_full, M_lower)
    return
    