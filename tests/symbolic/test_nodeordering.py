import pytest

from symfem.symbols import x

from symfem import create_element

from topoptlab.symbolic.cell import base_cell, node_ordering

@pytest.mark.parametrize('cell_name,nd_inds',
                         [("interval", [0, 1]),
                          ("quadrilateral", [0, 1, 3, 2]),
                          ("hexahedron", [0, 1, 3, 2, 4, 5, 7, 6])])
def test_node_ordering_linear(cell_name, nd_inds):
    """
    Ordering of linear elements must not change.
    """
    assert node_ordering(create_element(cell_name, "Lagrange", 1))[0] == nd_inds
    return

def test_node_ordering_abaqus():
    """
    Quadratic elements follow the ABAQUS numbering of CPS8 and C3D20.
    """
    # CPS8 + center node
    coords = node_ordering(create_element("quadrilateral", "Lagrange", 2))[1]
    assert coords == ((-1, -1), (1, -1), (1, 1), (-1, 1),
                      (0, -1), (1, 0), (0, 1), (-1, 0),
                      (0, 0))
    # C3D20: corners, bottom edges, top edges, vertical edges
    coords = node_ordering(create_element("hexahedron", "Lagrange", 2))[1]
    assert coords[:20] == ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                           (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                           (0, -1, -1), (1, 0, -1), (0, 1, -1), (-1, 0, -1),
                           (0, -1, 1), (1, 0, 1), (0, 1, 1), (-1, 0, 1),
                           (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
    assert coords[-1] == (0, 0, 0)
    return

@pytest.mark.parametrize('ndim,order',
                         [(1, 1), (1, 2), (1, 3),
                          (2, 1), (2, 2), (2, 3),
                          (3, 1), (3, 2)])
def test_base_cell_kronecker(ndim, order):
    """
    Basis function i must be 1 at node i and 0 at all other nodes.
    """
    vertices, nd_inds, ref, basis = base_cell(ndim=ndim, order=order)
    assert len(vertices) == len(basis) == (order+1)**ndim
    for i, b in enumerate(basis):
        for j, p in enumerate(vertices):
            assert b.subs(x[:ndim], p).as_sympy() == (1 if i == j else 0)
    return

@pytest.mark.parametrize('cell_name,element_type,order',
                         [("quadrilateral", "serendipity", 2),
                          ("quadrilateral", "NCE", 1),
                          ("hexahedron", "Brezzi-Douglas-Duran-Fortin", 1),
                          ("interval", "Hermite", 3),
                          ("quadrilateral", "vector Q", 1),
                          ("quadrilateral", "direct serendipity", 1)])
def test_node_ordering_nonnodal(cell_name, element_type, order):
    """
    Elements that are not nodal must be rejected.
    """
    with pytest.raises(NotImplementedError):
        node_ordering(create_element(cell_name, element_type, order))
    return
