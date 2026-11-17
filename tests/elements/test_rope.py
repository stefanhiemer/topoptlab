from numpy import array, zeros, stack, sqrt, ix_, arange, concatenate
from numpy.linalg import solve
from numpy.testing import assert_allclose, assert_equal

import pytest

from topoptlab.fem import append_multipliers
from topoptlab.elements.bar import bar_geometry, create_edofMat
from topoptlab.elements.bar_1d import lk_bar_1d
from topoptlab.elements.bar_2d import lk_bar_2d
from topoptlab.elements.bar_3d import lk_bar_3d
from topoptlab.elements.rope_1d import lk_rope_1d
from topoptlab.elements.rope_2d import lk_rope_2d
from topoptlab.elements.rope_3d import lk_rope_3d

coords = array([[0., 0., 0.], [1., 0., 0.], [1., 2., 0.], [0.5, 1., 3.]])
bars = array([[0, 1], [1, 2], [2, 3], [0, 3]])


@pytest.mark.parametrize('lk_rope,lk_bar,ndim', [(lk_rope_1d, lk_bar_1d, 1),
                                                 (lk_rope_2d, lk_bar_2d, 2),
                                                 (lk_rope_3d, lk_bar_3d, 3)])
def test_schur(lk_rope, lk_bar, ndim):
    # eliminating the force multiplier recovers the bar stiffness
    if ndim == 1:
        l, kw = array([1., 1.5, 2., 0.3]), {}
    else:
        l, n = bar_geometry(coords[:, :ndim], bars)
        kw = {"n": n}
    E, A = array([1., 2., 3., 4.]), 0.5
    Ke = lk_rope(E=E, A=A, l=l, **kw)
    k = 2*ndim
    assert Ke.shape == (4, k+1, k+1)
    assert_allclose(Ke, Ke.transpose(0, 2, 1))
    assert_allclose(Ke[:, :k, :k], 0.)
    schur = - Ke[:, :k, k:] @ Ke[:, k:, :k] / Ke[:, k:, k:]
    assert_allclose(schur, lk_bar(E=E, A=A, l=l, **kw))
    # batched equals single calls
    ref = stack([lk_rope(E=E[i], A=A, l=l[i],
                         **({k_: v[i] for k_, v in kw.items()}))[0]
                 for i in range(4)])
    assert_allclose(Ke, ref)


def test_append_multipliers():
    edofMat = create_edofMat(bars, 2)
    ext = append_multipliers(edofMat, ndof=8)
    assert_equal(ext[:, :4], edofMat)
    assert_equal(ext[:, 4], 8 + arange(4))
    ext = append_multipliers(edofMat, ndof=8, n_mult=2)
    assert_equal(ext[:, 4:], 8 + arange(8).reshape(4, 2))


@pytest.mark.parametrize('P', [1., -1.])
def test_truss_mixed(P):
    # symmetric two-bar truss: supports at (-1,0),(1,0), apex (0,1) loaded
    # vertically. P>0 pulls apex up (tension), P<0 pushes down (compression)
    coords2 = array([[-1., 0.], [1., 0.], [0., 1.]])
    bars2 = array([[0, 2], [1, 2]])
    E, A = 10., 0.5
    l, n = bar_geometry(coords2, bars2)
    # standard displacement formulation
    Kb = lk_bar_2d(E=E, A=A, l=l, n=n)
    edofMat = create_edofMat(bars2, 2)
    K = zeros((6, 6))
    for e in range(2):
        K[ix_(edofMat[e], edofMat[e])] += Kb[e]
    free = array([4, 5])
    f = zeros(6)
    f[5] = P
    u_ref = solve(K[ix_(free, free)], f[free])
    # mixed formulation with appended force multipliers
    Ke = lk_rope_2d(E=E, A=A, l=l, n=n)
    edofMat = append_multipliers(edofMat, ndof=6)
    S = zeros((8, 8))
    for e in range(2):
        S[ix_(edofMat[e], edofMat[e])] += Ke[e]
    free = concatenate((free, [6, 7]))
    rhs = concatenate((f, zeros(2)))
    x = solve(S[ix_(free, free)], rhs[free])
    assert_allclose(x[:2], u_ref, atol=1e-12)
    # bar forces: N = P/(2 sin45), sign gives tension/compression
    assert_allclose(x[2:], P/(2/sqrt(2)) * array([1., 1.]))
