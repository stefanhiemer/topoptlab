from numpy import array
from numpy.testing import assert_equal,assert_allclose
from scipy.sparse import csc_array

from topoptlab.amg import rubestueben_coupling

import pytest

@pytest.mark.parametrize('test, sol_mask, sol_s, sol_s_t, sol_iso',
                         [(array([[1., -0.25, -1., 0.55, 0.1, 0.],
                                  [-0.25, 1., 0., 0., 0., 0.],
                                  [-1., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., -0.1, -2.2, 1., 0], 
                                  [0., 0., 0., 0., 0., 1.]]), 
                           array([True, True, True, False, 
                                  True, 
                                  True, True, False, 
                                  False, True, True, 
                                  False, False, True]), 
                           [[1,2,3],
                            [0],
                            [0,3],
                            [2,4],
                            [3],
                            []], 
                           [[1,2],
                            [0],
                            [0,3],
                            [0,2,4],
                            [3], 
                            []],
                           [5]),
                          (array([[1., 0., -0.25, -1., 0.55, 0.1],
                                  [0., 1., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1],
                                  [0.55, 0., 0., -1.2, 5, -2.2],
                                  [0.1, 0., 0., -0.1, -2.2, 1.]]), 
                            array([True, True, True, False, 
                                   True, 
                                   True, True, False, 
                                   False, True, True, 
                                   False, False, True]), 
                            [[2,3,4],
                             [],
                             [0],
                             [0,4],
                             [3,5],
                             [4]], 
                            [[2,3],
                             [],
                             [0],
                             [0,4],
                             [0,3,5],
                             [4]],
                            [1]),
                          (array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1., 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]] ), 
                            array([True, True, True, False, 
                                   True, 
                                   True, True, False, 
                                   False, True, True, 
                                   False, False, True]), 
                            [[2,3,4],
                             [],
                             [0],
                             [0,4],
                             [3,5],
                             [4],
                             []], 
                            [[2,3],
                             [],
                             [0],
                             [0,4],
                             [0,3,5],
                             [4], 
                             []],
                            [1,6])])

def test_rubestuebgen_coupling(test, 
                               sol_mask, sol_s, sol_s_t, sol_iso):
    
    #
    test = csc_array(test)
    _,_,mask_strong,s,s_t,iso = rubestueben_coupling(A=test, 
                                                     c_neg = 0.2, 
                                                     c_pos = 0.5)
    # for testing sort it
    s = [sorted(entry) for entry in s]
    s_t = [sorted(entry) for entry in s_t]
    #
    assert_equal(sol_mask, mask_strong)
    #
    for sol,actual in zip(sol_s,s):
        assert_equal(actual, sol)
    #
    for sol,actual in zip(sol_s_t,s_t):
        assert_equal(actual, sol)
    #
    assert_equal(sol_iso, iso)
    return

from topoptlab.amg import standard_coarsening

@pytest.mark.parametrize('test, sol_mask',
                         [(array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1., 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]]), 
                          array([False, False, True, False, True, False, False]))])

def test_standard_coarsening(test, sol_mask):
    #
    test = csc_array(test)
    #
    mask_coarse = standard_coarsening(test,
                                      coupling_fnc=rubestueben_coupling,
                                      coupling_kw = {"c_neg": 0.2, 
                                                     "c_pos": 0.5})
    #
    assert_equal(sol_mask, mask_coarse)
    return

from topoptlab.amg import direct_interpolation

@pytest.mark.parametrize('test, solution',
                         [(array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                                  [0., 1., 0., 0., 0., 0., 0.],
                                  [-0.25, 0., 1., 0., 0., 0., 0.],
                                  [-1., 0., 0., 2., -1.2, -0.1, 0.],
                                  [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                                  [0.1, 0., 0., -0.1, -2.2, 1.7, 0.], 
                                  [0., 0., 0., 0., 0., 0., 1.]] ), 
                          array( [[1.25, -0.65],
                                  [0., 0.],
                                  [1., 0.],
                                  [0., 1.15],
                                  [0., 1.],
                                  [0., 1.35294118],
                                  [0., 0.]]))])

def test_direct_interpolation(test, solution):
    
    
    #
    test = csc_array(test)
    #
    mask_coarse = standard_coarsening(test,
                                      coupling_fnc=rubestueben_coupling,
                                      coupling_kw = {"c_neg": 0.2, 
                                                     "c_pos": 0.5})
    #
    P = direct_interpolation(test, mask_coarse)
    #
    assert_allclose(P.toarray(),solution)
    return

from numpy import arange,array,ones,prod
from scipy.sparse import coo_array

from topoptlab.amg import rigid_bodymodes
from topoptlab.fem import create_matrixinds
from topoptlab.utils import nodeid_to_coords

@pytest.mark.parametrize('nelx, nely, nelz',
                         [(2,2,None),
                          (10,3,None),
                          (2,2,2),
                          (10,3,5),])

def test_rigid_modes(nelx,nely,nelz):
    
    #
    if nelz is None:
        ndim = 2
        from topoptlab.elements.bilinear_quadrilateral import create_edofMat
        from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d as lk
    else:
        ndim = 3
        from topoptlab.elements.trilinear_hexahedron import create_edofMat
        from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d as lk
    # total number of design elements
    n = prod([nelx,nely,nelz][:ndim])
    ndof = ndim*prod( array([nelx,nely,nelz][:ndim])+1)
    # Max and min stiffness
    Emin=1e-9
    Emax=1.0
    # fetch element stiffness matrix
    KE = lk()
    # dofs:
    n_ndof = int(KE.shape[-1]/2**ndim)
    ndof = n_ndof * prod( array([nelx,nely,nelz][:ndim])+1 )
    # FE: Build the index vectors for the for coo matrix format.
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,nely=nely,nelz=nelz,
                                             nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat, mode="full")
    # Setup and solve FE problem
    sK=(KE.flatten()[:,None]*(Emin+ones(n)*(Emax-Emin))).flatten(order='F')
    K = coo_array((sK,(iK,jK)),shape=(ndof,ndof)).asformat("csr")
    #
    coords = nodeid_to_coords(nd = arange( prod( array([nelx,
                                                        nely,
                                                        nelz][:ndim])+1) ),
                              nelx=nelx, 
                              nely=nely,
                              nelz=nelz)
    #
    u_b = rigid_bodymodes(coords)
    #
    assert_allclose((K@u_b).max(), 0., atol=1e-14)
    return


