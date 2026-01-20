from numpy import array,stack,vstack,zeros
from numpy.testing import assert_allclose

import pytest

from topoptlab.elements.check_functions import check_inputs


@pytest.mark.parametrize("coords,n_true",    
                         [((0.0,), 1),
                          ((array([0.0, 0.5, 1.0]),), 3),
                          ],)

def test_scalar_coords_no_xe(coords, n_true):
    nnodes = 2
    ndim = len(coords)
    n = check_inputs(*coords, ndim=ndim, nnodes=nnodes)
    assert n == n_true


@pytest.mark.parametrize("coords",
                         [(0.0, array([0.1])),
                          ],)

def test_mixed_coord_types_raises(coords):
    with pytest.raises(ValueError, match="coordinate datatypes are inconsistent"):
        check_inputs(*coords, ndim=2, nnodes=4)


@pytest.mark.parametrize("coords",
                         [(array([1.0, 0.1]), array([0.1])),
                          ],)

def test_inconsistent_coord_lengths_raises(coords):
    with pytest.raises(ValueError, match="all coordinates must have the same shape"):
        check_inputs(*coords, ndim=2, nnodes=4)

@pytest.mark.parametrize("coords,nnodes,ndim",
                         [((0.0,), 2, 1),
                          ((0.0, 0.0), 4, 2),
                          ((0.0, 0.0, 0.0), 8, 3),
                          ],)

def test_padding_and_unpacking(coords, nnodes, ndim):
    #
    xe = zeros((1, nnodes, ndim))
    xe_out, xi, eta, zeta = check_inputs(*coords, 
                                         ndim=ndim, 
                                         nnodes=nnodes, 
                                         xe=xe)

    assert xe_out.shape == (1, nnodes, ndim)
    if ndim >= 1:
        assert_allclose(xi,coords[0])
    if ndim >= 2:
        assert_allclose(eta,coords[1])
    if ndim == 3:
        assert_allclose(zeta,coords[2])
    if ndim == 1:
        assert eta is None
        assert zeta is None
    elif ndim == 2:
        assert zeta is None

@pytest.mark.parametrize("xe,nnodes,ndim",
                         [(zeros((4, 2)), 4, 2),
                          (zeros((8, 3)), 8, 3),
                          ],)

def test_xe_2d_input_promoted(xe, nnodes, ndim):
    coords = tuple(0.0 for _ in range(ndim))
    xe_out, *_ = check_inputs(*coords, 
                              ndim=ndim, 
                              nnodes=nnodes, 
                              xe=xe)
    assert xe_out.shape == (1, nnodes, ndim)


@pytest.mark.parametrize("xe,nnodes,ndim",
                         [(zeros((4, 3)), 4, 2),
                          (zeros((8, 2)), 8, 3),
                          ],)

def test_xe_wrong_shape_raises(xe, nnodes, ndim):
    coords = tuple(0.0 for _ in range(ndim))
    with pytest.raises(ValueError, match="xe must have shape"):
        check_inputs(*coords, ndim=ndim, nnodes=nnodes, xe=xe)

@pytest.mark.parametrize("coords,xe_shape,expected_len",
                         [((array([0.0, 0.5]), 
                            array([0.0, 0.5])), 
                           (2, 4, 2), 2),
                          ((array([0.0, 0.5, 1.0, 1.5]), 
                            array([0.0, 0.5, 1.0, 1.5])), 
                           (4, 4, 2), 4),],)

def test_single_element_multiple_coords(coords, xe_shape, expected_len):
    xe = zeros(xe_shape)
    xe_out, xi, eta, _ = check_inputs(*coords, ndim=2, nnodes=4, xe=xe)

    assert xi.shape[0] == expected_len
    assert eta.shape[0] == expected_len
    if expected_len == xe_shape[0]:
        assert xe_out.shape == xe_shape
    else:
        assert xe_out.shape == (expected_len, *xe_shape[1:])


@pytest.mark.parametrize("coords,xe_shape",
                         [((array([0, 1, 2, 3, 4, 5, 6, 7]),
                            array([0, 1, 2, 3, 4, 5, 6, 7])), 
                           (2, 4, 2)), 
                          ],)

def test_nnodes_times_nels_repetition(coords, xe_shape):
    xe = zeros(xe_shape)
    xe_out, xi, eta, _ = check_inputs(*coords, ndim=2, nnodes=4, xe=xe)

    assert xe_out.shape[0] == 8
    assert xi.shape[0] == 8
    assert eta.shape[0] == 8


@pytest.mark.parametrize("coords,xe_shape",
                         [((array([0.0, 0.5, 1.0]), 
                            array([0.0, 0.5, 1.0])), 
                           (2, 4, 2)),
                          ],)

def test_incompatible_shapes_raise(coords, xe_shape):
    xe = zeros(xe_shape)
    with pytest.raises(ValueError, match="shapes of nels and ncoords incompatible"):
        check_inputs(*coords, ndim=2, nnodes=4, xe=xe)

@pytest.mark.parametrize("coords,xe_shape",
                         [((array([0.0, 0.5]), 
                            array([0.0, 0.5])), 
                           (3, 4, 2)),
                          ],)

def test_all_elems_tiles_coords(coords, xe_shape):
    xe = zeros(xe_shape)
    xe_out, xi, eta, _ = check_inputs(*coords, 
                                      ndim=2, 
                                      nnodes=4, 
                                      xe=xe, 
                                      all_elems=True)

    assert xe_out.shape[0] == 6
    assert xi.shape[0] == 6
    assert eta.shape[0] == 6 

def test_ndim_greater_than_3_raises():
    with pytest.raises(ValueError, match="ndim must be <= 3"):
        check_inputs(0.0, 0.0, 0.0, 0.0, ndim=4, nnodes=16)