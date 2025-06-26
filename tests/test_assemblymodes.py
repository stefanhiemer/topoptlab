from numpy import lexsort,column_stack,tril_indices_from,prod,tril
from numpy.random import seed, rand
from numpy.testing import assert_almost_equal

import pytest

from topoptlab.fem import assemble_matrix,create_matrixinds

@pytest.mark.parametrize('nelx, nely, nelz, n_ndof, assembly_mode',
                         [(10,3,None,1,"lower"),
                          (10,3,None,2,"lower"),
                          (10,3,3,1,"lower"),
                          (10,3,3,3,"lower"),])

def test_assembly(nelx, nely, nelz, n_ndof, assembly_mode):
    """
    Checks that different assembly modes yield same assembled matrix.
    """
    #
    seed(0)
    #
    if nelz is None:
        n = nelx*nely
        ndof = (nelx+1)*(nely+1)*n_ndof
        ndim = 2
        from topoptlab.elements.bilinear_quadrilateral import create_edofMat
    else:
        n = nelx*nely*nelz
        ndof = (nelx+1)*(nely+1)*(nelz+1)*n_ndof
        ndim = 3
        from topoptlab.elements.trilinear_hexahedron import create_edofMat
    #
    edofMat = create_edofMat(nelx=nelx,nely=nely,nelz=nelz,nnode_dof=n_ndof)[0]
    #
    _iK,_jK = create_matrixinds(edofMat,mode="full")
    iK,jK = create_matrixinds(edofMat,mode="lower")
    #
    KE = rand(n,2**ndim * n_ndof, 2**ndim * n_ndof)
    KE = KE + KE.transpose(0,2,1)
    #
    assm_indcs = column_stack(tril_indices_from(KE[0]))
    assm_indcs = assm_indcs[lexsort( (assm_indcs[:,0],assm_indcs[:,1]) )]
    #
    _sK = KE.reshape(prod(KE.shape))
    sK = KE[:,assm_indcs[:,0],assm_indcs[:,1]].reshape( n*int(KE.shape[-1]/2*(KE.shape[-1]+1)) )
    # assemble stiffness matrices
    _K = assemble_matrix(sK=_sK,iK=_iK,jK=_jK,
                        ndof=ndof,solver="scipy-direct",
                        springs=None)
    K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                        ndof=ndof,solver="scipy-direct",
                        springs=None)
    #
    assert_almost_equal(tril(_K.todense()) , K.todense())
    return
