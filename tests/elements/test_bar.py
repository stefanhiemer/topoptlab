from numpy import array, eye, zeros, outer, cross, stack, sqrt, ix_
from numpy.linalg import eigvalsh, norm, solve
from numpy.testing import assert_allclose, assert_equal

import pytest

from topoptlab.elements.bar import bar_geometry, create_edofMat
from topoptlab.elements.bar_1d import lk_bar_1d
from topoptlab.elements.bar_2d import lk_bar_2d
from topoptlab.elements.bar_3d import lk_bar_3d


@pytest.mark.parametrize('E,A,l', [(1., 1., 1.), (210., 0.3, 2.5)])
def test_1d(E, A, l):
    Ke = lk_bar_1d(E=E, A=A, l=l)
    assert Ke.shape == (1, 2, 2)
    Ke = Ke[0]
    assert_allclose(Ke, Ke.T)
    assert_allclose(Ke @ array([1., 1.]), zeros(2), atol=1e-12)
    assert_allclose(eigvalsh(Ke), [0., 2*E*A/l], atol=1e-12)


@pytest.mark.parametrize('n', [array([1., 0.]), array([0., 1.]),
                               array([1., 1.]), array([-3., 4.])])
def test_2d(n):
    E, A, l = 70., 0.2, 1.7
    Ke = lk_bar_2d(E=E, A=A, l=l, n=n)
    assert Ke.shape == (1, 4, 4)
    Ke = Ke[0]
    assert_allclose(Ke, Ke.T)
    # rigid body modes: two translations and in-plane rotation
    nu = n / norm(n)
    rbm = [array([1., 0., 1., 0.]),
           array([0., 1., 0., 1.]),
           array([0., 0., -nu[1], nu[0]])]
    for r in rbm:
        assert_allclose(Ke @ r, zeros(4), atol=1e-12)
    assert_allclose(eigvalsh(Ke), [0., 0., 0., 2*E*A/l], atol=1e-12)
    # axial stretch reproduces 1D bar
    u = array([0., 0., *nu])
    assert_allclose(u @ Ke @ u, lk_bar_1d(E=E, A=A, l=l)[0, 1, 1])


@pytest.mark.parametrize('n', [array([1., 0., 0.]), array([0., 0., 1.]),
                               array([1., 1., 1.]), array([-1., 2., 0.5])])
def test_3d(n):
    E, A, l = 70., 0.2, 1.7
    Ke = lk_bar_3d(E=E, A=A, l=l, n=n)
    assert Ke.shape == (1, 6, 6)
    Ke = Ke[0]
    assert_allclose(Ke, Ke.T)
    # rigid body translations
    for i in range(3):
        t = zeros(6)
        t[[i, i+3]] = 1.
        assert_allclose(Ke @ t, zeros(6), atol=1e-12)
    # rotations about axes perpendicular to bar (node 1 at origin)
    nu = n / norm(n)
    for w in eye(3):
        r = zeros(6)
        r[3:] = cross(w, nu)
        assert_allclose(Ke @ r, zeros(6), atol=1e-12)
    assert_allclose(eigvalsh(Ke), [0.]*5 + [2*E*A/l], atol=1e-12)
    # closed form equals block outer product
    N = outer(nu, nu)
    ref = E*A/l * array(((N, -N), (-N, N))).transpose(0, 2, 1, 3).reshape(6, 6)
    assert_allclose(Ke, ref, atol=1e-12)


@pytest.mark.parametrize('n2', [array([1., 0.]), array([3., 4.]),
                                array([-1., 1.])])
def test_2d_vs_3d(n2):
    # in-plane 3D bar must reduce to 2D bar
    K2 = lk_bar_2d(E=2., A=0.5, l=3., n=n2)[0]
    K3 = lk_bar_3d(E=2., A=0.5, l=3., n=array([*n2, 0.]))[0]
    idx = [0, 1, 3, 4]
    assert_allclose(K3[ix_(idx, idx)], K2, atol=1e-12)


def test_1d_vs_2d():
    K1 = lk_bar_1d(E=2., A=0.5, l=3.)[0]
    K2 = lk_bar_2d(E=2., A=0.5, l=3., n=array([1., 0.]))[0]
    assert_allclose(K2[ix_([0, 2], [0, 2])], K1)
    # direction need not be normalized
    assert_allclose(lk_bar_2d(n=array([3., 4.])),
                    lk_bar_2d(n=array([3., 4.])/5.))


@pytest.mark.parametrize('lk,ndim', [(lk_bar_1d, 1),
                                     (lk_bar_2d, 2),
                                     (lk_bar_3d, 3)])
def test_vectorized(lk, ndim):
    # batched call must equal stacked single-element calls
    coords = array([[0., 0., 0.], [1., 0., 0.], [1., 2., 0.],
                    [0.5, 1., 3.]])[:, :ndim]
    if ndim == 1:
        coords = array([[0.], [1.], [2.5], [4.]])
    bars = array([[0, 1], [1, 2], [2, 3], [0, 3]])
    l, n = bar_geometry(coords, bars)
    E = array([1., 2., 3., 4.])
    A = array([0.5, 0.1, 1., 2.])
    kw = {} if ndim == 1 else {"n": n}
    Ke = lk(E=E, A=A, l=l, **kw)
    assert Ke.shape == (4, 2*ndim, 2*ndim)
    if ndim == 1:
        ref = stack([lk(E=E[i], A=A[i], l=l[i])[0] for i in range(4)])
    else:
        ref = stack([lk(E=E[i], A=A[i], l=l[i], n=n[i])[0]
                     for i in range(4)])
    assert_allclose(Ke, ref)
    # scalar material, arrays of geometry
    if ndim > 1:
        assert_allclose(lk(E=2., A=0.5, l=l, n=n),
                        stack([lk(E=2., A=0.5, l=l[i], n=n[i])[0]
                               for i in range(4)]))


def test_geometry():
    coords = array([[0., 0.], [3., 4.], [3., 0.]])
    bars = array([[0, 1], [2, 1], [1, 2]])
    l, n = bar_geometry(coords, bars)
    assert_allclose(l, [5., 4., 4.])
    assert_allclose(n, [[0.6, 0.8], [0., 1.], [0., -1.]])
    with pytest.raises(ValueError):
        bar_geometry(coords, array([[0, 0]]))


def test_edofMat():
    bars = array([[0, 1], [2, 1], [3, 0]])
    assert_equal(create_edofMat(bars, 1), bars)
    assert_equal(create_edofMat(bars, 2),
                 [[0, 1, 2, 3], [4, 5, 2, 3], [6, 7, 0, 1]])
    assert_equal(create_edofMat(bars, 3),
                 [[0, 1, 2, 3, 4, 5], [6, 7, 8, 3, 4, 5],
                  [9, 10, 11, 0, 1, 2]])


def test_truss_2d():
    # symmetric two-bar truss: supports at (-1,0),(1,0), loaded apex (0,1)
    coords = array([[-1., 0.], [1., 0.], [0., 1.]])
    bars = array([[0, 2], [1, 2]])
    E, A, P = 10., 0.5, 1.
    l, n = bar_geometry(coords, bars)
    Ke = lk_bar_2d(E=E, A=A, l=l, n=n)
    edofMat = create_edofMat(bars, 2)
    K = zeros((6, 6))
    for e in range(bars.shape[0]):
        K[ix_(edofMat[e], edofMat[e])] += Ke[e]
    free = [4, 5]
    f = array([0., -P])
    u = solve(K[ix_(free, free)], f)
    # analytical: bar force N = P/(2 sin45), vertical disp = N l/(EA sin45)
    s = 1/sqrt(2)
    assert_allclose(u, [0., -P/(2*s) * sqrt(2)/(E*A*s)], atol=1e-12)
