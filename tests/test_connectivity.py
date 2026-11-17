import numpy as np
from numpy.testing import assert_equal

import pytest

from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d,\
                                                      apply_pbc
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d,\
                                                     apply_pbc as apply_pbc3d
from topoptlab.utils import map_imgtoel, map_eltoimg, element_adjacency,\
                            grid_adjacency
from topoptlab.design_analysis.connectivity import boundary_elements,\
                                                   phase_components,\
                                                   void_islands,\
                                                   solid_islands,\
                                                   graph_laplacian,\
                                                   spectral_connectivity

nelx, nely = 9, 7

def design(img):
    return map_imgtoel(np.array(img, dtype=float), nelx=nelx, nely=nely)

# ring with enclosed void (2 elements) and a floating solid element
ring = np.zeros((nely,nelx))
ring[1:6,1:5] = 1.
ring[2:4,2:4] = 0.
ring[3,7] = 1.

@pytest.mark.parametrize('nelz', [None, 4])
@pytest.mark.parametrize('pbc', [False, True])
@pytest.mark.parametrize('n_shared', [None, 1, 2])
def test_grid_adjacency(nelz, pbc, n_shared):
    if nelz is None:
        pbc = [pbc, False]
        el = create_edofMat2d(nelx=5, nely=3, nnode_dof=1)[0]
        if any(pbc):
            el = apply_pbc(el, pbc=pbc, nelx=5, nely=3, nnode_dof=1)
    else:
        pbc = [pbc, False, pbc]
        el = create_edofMat3d(nelx=5, nely=3, nelz=nelz, nnode_dof=1)[0]
        if any(pbc):
            el = apply_pbc3d(el, pbc=pbc, nelx=5, nely=3, nelz=nelz,
                             nnode_dof=1)
    A = grid_adjacency(nelx=5, nely=3, nelz=nelz, pbc=pbc, n_shared=n_shared)
    B = element_adjacency(el, n_shared=n_shared)
    assert (A != B).nnz == 0
    return

def test_adjacency():
    A = grid_adjacency(nelx=nelx, nely=nely)
    deg = map_eltoimg((A != 0).sum(axis=1), nelx=nelx, nely=nely)
    # 4-neighborhood
    assert deg[0,0] == 2 and deg[0,4] == 3 and deg[3,4] == 4
    assert (A != A.T).nnz == 0 and A.diagonal().sum() == 0
    # 8-neighborhood
    deg8 = map_eltoimg((grid_adjacency(nelx=nelx, nely=nely, n_shared=1) != 0).sum(axis=1),
                       nelx=nelx, nely=nely)
    assert deg8[0,0] == 3 and deg8[3,4] == 8
    # boundary elements
    bnd = map_eltoimg(boundary_elements(A, ndim=2), nelx=nelx, nely=nely)
    ref = np.ones((nely,nelx), dtype=bool)
    ref[1:-1,1:-1] = False
    assert_equal(bnd, ref)
    return

def test_islands():
    A = grid_adjacency(nelx=nelx, nely=nely)
    x = design(ring)
    n, labels = phase_components(x, A, phase="solid")
    assert n == 2
    n, _ = phase_components(x, A, phase="void")
    assert n == 2
    # enclosed void
    n, mask = void_islands(x, A, boundary=boundary_elements(A, ndim=2))
    ref = np.zeros((nely,nelx), dtype=bool)
    ref[2:4,2:4] = True
    assert n == 1
    assert_equal(map_eltoimg(mask, nelx=nelx, nely=nely), ref)
    # floating element (largest component is the structure)
    n, mask = solid_islands(x, A)
    ref = np.zeros((nely,nelx), dtype=bool)
    ref[3,7] = True
    assert n == 1
    assert_equal(map_eltoimg(mask, nelx=nelx, nely=nely), ref)
    # anchor at floating element: ring is island
    n, mask = solid_islands(x, A, anchor=mask)
    assert n == 1 and mask.sum() == ring.sum()-1
    # diagonal contact counts as connection for node adjacency
    x2 = ring.copy()
    x2[4,6] = 1.
    n, _ = solid_islands(design(x2), A)
    assert n == 2
    n, _ = solid_islands(design(x2), grid_adjacency(nelx=nelx, nely=nely, n_shared=1))
    assert n == 1
    return

def test_periodic():
    # vertical bar touching left/right boundary is enclosed only with pbc
    img = np.ones((nely,nelx))
    img[2:5,:] = 0.
    A = grid_adjacency(nelx=nelx, nely=nely, pbc=[True,False])
    bnd = boundary_elements(A, ndim=2)
    assert map_eltoimg(bnd, nelx=nelx, nely=nely)[3].sum() == 0
    n, _ = void_islands(design(img), A, boundary=bnd)
    assert n == 1
    A = grid_adjacency(nelx=nelx, nely=nely)
    n, _ = void_islands(design(img), A, boundary=boundary_elements(A, ndim=2))
    assert n == 0
    return

def test_hexahedron():
    nelz = 5
    A = grid_adjacency(nelx=5, nely=5, nelz=nelz)
    assert (A != 0).sum(axis=1).max() == 6
    # hollow cube
    x = np.zeros((nelz,5,5))
    x[1:4,1:4,1:4] = 1.
    x[2,2,2] = 0.
    # element ordering of regular grid: x fastest in y, then x, then z
    x = x.transpose((0,2,1)).reshape(-1)
    n, mask = void_islands(x, A, boundary=boundary_elements(A, ndim=3))
    assert n == 1 and mask.sum() == 1
    return

def test_laplacian():
    A = grid_adjacency(nelx=nelx, nely=nely)
    L = graph_laplacian(A, w=np.ones(A.nnz//2))
    assert np.allclose(L.sum(axis=1), 0.)
    assert np.allclose(L.toarray() + A.toarray(),
                       np.diag((A != 0).sum(axis=1)))
    return

@pytest.mark.parametrize('phase', ["solid", "void"])
def test_spectral(phase):
    A = grid_adjacency(nelx=nelx, nely=nely)
    x = design(ring)
    if phase == "solid":
        fixed = np.zeros(x.shape[0], dtype=bool)
        fixed[np.flatnonzero(design(ring == 1))[0]] = True
        n_ref = 1
    else:
        fixed = boundary_elements(A, ndim=2)
        n_ref = 1
    lam = spectral_connectivity(x, A, fixed=fixed, phase=phase,
                                w_min=1e-6, m_min=1e-9, k=4)
    assert np.all(np.diff(lam) >= 0)
    # islands give eigenvalues of order w_min, rest of order 1e-2 or larger
    assert (lam < 1e-3).sum() == n_ref
    # connected design: no small eigenvalue
    x_full = np.ones_like(x) if phase == "solid" else np.zeros_like(x)
    lam = spectral_connectivity(x_full, A, fixed=fixed, phase=phase,
                                w_min=1e-6, m_min=1e-9, k=4)
    assert (lam < 1e-3).sum() == 0
    return
