from numpy import array, isclose
from numpy.testing import assert_allclose, assert_equal

import pytest

gmsh = pytest.importorskip("gmsh")

from topoptlab.geometry_parser import read_mesh, element_jacobian_det
from topoptlab.topology_opt.mesh import create_mesh

def create_rect(mesh_file, nelx=4, nely=2, z=0., recombine=True):
    # transfinite rectangle [0,nelx]x[0,nely] at height z
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    tag = gmsh.model.occ.addRectangle(0., 0., z, nelx, nely)
    gmsh.model.occ.synchronize()
    for _, c in gmsh.model.getBoundary([(2, tag)], oriented=False):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, c)
        if abs(ymax - ymin) < 1e-6:
            gmsh.model.mesh.setTransfiniteCurve(c, nelx + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(c, nely + 1)
            if xmax < 1e-6:
                pg = gmsh.model.addPhysicalGroup(1, [c])
                gmsh.model.setPhysicalName(1, pg, "left")
    gmsh.model.mesh.setTransfiniteSurface(tag)
    if recombine:
        gmsh.model.mesh.setRecombine(2, tag)
    pg = gmsh.model.addPhysicalGroup(2, [tag])
    gmsh.model.setPhysicalName(2, pg, "domain")
    gmsh.model.mesh.generate(2)
    gmsh.write(str(mesh_file))
    gmsh.finalize()
    return

def create_box(mesh_file, n=2):
    # transfinite unit box with n elements per direction
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    tag = gmsh.model.occ.addBox(0., 0., 0., 1., 1., 1.)
    gmsh.model.occ.synchronize()
    for _, c in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(c, n + 1)
    for _, s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    gmsh.model.mesh.setTransfiniteVolume(tag)
    gmsh.model.mesh.setRecombine(3, tag)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(mesh_file))
    gmsh.finalize()
    return

def create_truss(mesh_file):
    # two bars in the plane z=1 meeting at the apex
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    p = [gmsh.model.geo.addPoint(*x, 1.) for x in [(-1., 0.), (1., 0.), (0., 1.)]]
    l1 = gmsh.model.geo.addLine(p[0], p[2])
    l2 = gmsh.model.geo.addLine(p[1], p[2])
    gmsh.model.geo.synchronize()
    for c in (l1, l2):
        gmsh.model.mesh.setTransfiniteCurve(c, 2)
    pg = gmsh.model.addPhysicalGroup(1, [l1, l2])
    gmsh.model.setPhysicalName(1, pg, "bars")
    pg = gmsh.model.addPhysicalGroup(0, [p[0], p[1]])
    gmsh.model.setPhysicalName(0, pg, "supports")
    gmsh.model.mesh.generate(1)
    gmsh.write(str(mesh_file))
    gmsh.finalize()
    return

def test_quad(tmp_path):
    f = tmp_path / "rect.msh"
    create_rect(f)
    mesh = read_mesh(str(f))
    assert mesh["element_type"] == "quadrilateral"
    assert mesh["coords"].shape == (15, 2)
    assert mesh["elnodes"].shape == (8, 4)
    detJ, w = element_jacobian_det(mesh["coords"][mesh["elnodes"]],
                                   "quadrilateral")
    assert (detJ > 0).all()
    assert_allclose((detJ*w).sum(), 8.)
    assert_equal(mesh["el_groups"]["domain"].shape[0], 8)
    left = mesh["node_groups"]["left"]
    assert left.shape[0] == 3
    assert isclose(mesh["coords"][left, 0], 0.).all()

def test_quad_offset_plane(tmp_path):
    # 2D mesh not lying in z=0 must still be detected as 2D
    f = tmp_path / "rect.msh"
    create_rect(f, z=2.)
    assert read_mesh(str(f))["coords"].shape[1] == 2
    assert read_mesh(str(f), ndim=3)["coords"].shape[1] == 3

def test_hex(tmp_path):
    f = tmp_path / "box.msh"
    create_box(f)
    mesh = read_mesh(str(f))
    assert mesh["element_type"] == "hexahedron"
    assert mesh["coords"].shape == (27, 3)
    assert mesh["elnodes"].shape == (8, 8)
    detJ, w = element_jacobian_det(mesh["coords"][mesh["elnodes"]],
                                   "hexahedron")
    assert (detJ > 0).all()
    assert_allclose((detJ*w).sum(), 1.)

def test_truss(tmp_path):
    f = tmp_path / "truss.msh"
    create_truss(f)
    mesh = read_mesh(str(f))
    assert mesh["element_type"] == "interval"
    # lies in z=1 plane, so 2D is inferred
    assert mesh["coords"].shape == (3, 2)
    assert_allclose(read_mesh(str(f), ndim=3)["coords"][:, 2], 1.)
    assert mesh["elnodes"].shape == (2, 2)
    supports = mesh["node_groups"]["supports"]
    assert_allclose(sorted(mesh["coords"][supports, 0]), [-1., 1.])

def test_triangles_rejected(tmp_path):
    f = tmp_path / "tri.msh"
    create_rect(f, recombine=False)
    with pytest.raises(ValueError):
        read_mesh(str(f))

def test_flip_and_unused(tmp_path):
    # single clockwise quad plus unused node 5
    f = tmp_path / "flip.msh"
    f.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                 "$Nodes\n5\n1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n5 7 7 0\n"
                 "$EndNodes\n"
                 "$Elements\n1\n1 3 2 0 1 1 4 3 2\n$EndElements\n")
    mesh = read_mesh(str(f))
    assert mesh["coords"].shape == (4, 2)
    detJ, w = element_jacobian_det(mesh["coords"][mesh["elnodes"]],
                                   "quadrilateral")
    assert (detJ > 0).all()
    assert_allclose((detJ*w).sum(), 2.)

def test_create_mesh(tmp_path):
    f = tmp_path / "rect.msh"
    create_rect(f)
    mesh = create_mesh(nelx=None, nely=None, nelz=None, l=0.5,
                       mesh_file=str(f))
    assert not mesh["regular"]
    assert mesh["ndim"] == 2 and mesh["n_el"] == 8 and mesh["n_nodes"] == 15
    assert_allclose(mesh["cellVolume"], 2.)
    assert_allclose(mesh["coords"].max(axis=0), array([2., 1.]))
    with pytest.raises(ValueError):
        create_mesh(nelx=None, nely=None, nelz=None, l=array([1., 1.]),
                    mesh_file=str(f))
    # regular mesh unchanged apart from flag
    assert create_mesh(nelx=4, nely=2, nelz=None)["regular"]
