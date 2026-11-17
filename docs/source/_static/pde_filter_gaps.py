# SPDX-License-Identifier: GPL-3.0-or-later
"""
Illustrate how convolution (density) filters and Helmholtz PDE filters behave
next to a gap in the design domain, similar to Fig. 1 (right) of

Lazarov, Boyan Stefanov, and Ole Sigmund. "Filters in topology optimization
based on Helmholtz-type differential equations." International journal for
numerical methods in engineering 86.6 (2011): 765-781.

The domain is a rectangle with a thin slit, meshed with irregular
quadrilaterals by gmsh. A unit density is placed in a single element next to
the slit and both filters are applied:

- convolution filter with hat kernel max(0, R - |x_i - x_j|): the filter
  reaches straight across the slit into material that is not connected.
- Helmholtz filter -r^2 Laplace(rho_f) + rho_f = rho with homogeneous Neumann
  boundary conditions: the slit is a boundary, so the response has to travel
  around the tip of the slit and practically nothing arrives on the other side.

The length scales are matched via R = 2*sqrt(3)*r (Eq. 34 of the paper).
"""
from os.path import dirname, abspath, join
from tempfile import TemporaryDirectory

import numpy as np
import gmsh
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from topoptlab.geometry_parser import read_mesh
from topoptlab.elements.poisson import _lk_poisson
from topoptlab.elements.mass_scalar import _lm_mass

# geometry: domain [0,LX]x[0,LY] with a slit of width SLIT_W at x=SLIT_X from
# y=SLIT_Y0 to the top edge
LX, LY = 1.5, 1.0
SLIT_X, SLIT_W, SLIT_Y0 = 0.75, 0.03, 0.3
# mesh size far from (LC_MAX) and close to (LC_MIN) the slit
LC_MAX, LC_MIN = 0.1, 0.02
# filter radius of convolution filter and equivalent Helmholtz length scale
R = 0.3
r = R / (2 * np.sqrt(3))
# location close to which the unit density is placed (left of the slit)
SOURCE = np.array([SLIT_X - 0.06, 0.65])
# lower limit of the log color scale
VMIN = 1e-3


def create_mesh(meshfile: str) -> None:
    """
    Mesh the slit domain with irregular quadrilaterals and write to file.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("slit")
    domain = gmsh.model.occ.addRectangle(0, 0, 0, LX, LY)
    slit = gmsh.model.occ.addRectangle(SLIT_X - SLIT_W / 2, SLIT_Y0, 0,
                                       SLIT_W, LY - SLIT_Y0)
    gmsh.model.occ.cut([(2, domain)], [(2, slit)])
    gmsh.model.occ.synchronize()
    # graded mesh size: fine at the slit, coarse far away
    slit_pt = gmsh.model.occ.addPoint(SLIT_X, SLIT_Y0, 0)
    gmsh.model.occ.synchronize()
    dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist, "PointsList", [slit_pt])
    thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh, "InField", dist)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMin", LC_MIN)
    gmsh.model.mesh.field.setNumber(thresh, "SizeMax", LC_MAX)
    gmsh.model.mesh.field.setNumber(thresh, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(thresh, "DistMax", 0.8)
    gmsh.model.mesh.field.setAsBackgroundMesh(thresh)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # unstructured all-quad mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(2)
    gmsh.write(meshfile)
    gmsh.finalize()
    return


def element_areas_centroids(xe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Area (shoelace formula) and centroid of each quadrilateral.
    """
    x, y = xe[..., 0], xe[..., 1]
    area = 0.5 * np.abs((x * np.roll(y, -1, axis=1)
                         - np.roll(x, -1, axis=1) * y).sum(axis=1))
    return area, xe.mean(axis=1)


def convolution_filter(centroids: np.ndarray, areas: np.ndarray,
                       rho: np.ndarray) -> np.ndarray:
    """
    Classic density filter with hat kernel based on Euclidean distance.
    """
    dist = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :],
                          axis=-1)
    w = np.maximum(0., R - dist) * areas[None, :]
    return (w @ rho) / w.sum(axis=1)


def helmholtz_filter(coords: np.ndarray, elnodes: np.ndarray,
                     rho: np.ndarray) -> np.ndarray:
    """
    Helmholtz filter on an arbitrary quadrilateral mesh. Homogeneous Neumann
    boundary conditions are fulfilled naturally by the weak form.
    """
    xe = coords[elnodes]
    nel, nnel = elnodes.shape
    # element matrices r^2 * Laplace + mass
    KE = _lk_poisson(xe=xe, k=r**2 * np.eye(2)) + _lm_mass(xe=xe)
    iK = np.repeat(elnodes, nnel, axis=1).flatten()
    jK = np.tile(elnodes, (1, nnel)).flatten()
    KF = coo_matrix((KE.flatten(), (iK, jK)),
                    shape=(coords.shape[0],) * 2).tocsc()
    # element density -> nodal right hand side: int N_i dOmega
    TE = _lm_mass(xe=xe).sum(axis=2)
    TF = coo_matrix((TE.flatten(),
                     (elnodes.flatten(), np.repeat(np.arange(nel), nnel))),
                    shape=(coords.shape[0], nel)).tocsc()
    # solve and average nodal values back to elements
    rho_nodal = spsolve(KF, TF @ rho)
    return rho_nodal[elnodes].mean(axis=1)


def plot_field(ax, xe: np.ndarray, field: np.ndarray, source: int,
               title: str) -> None:
    """
    Plot element field on the mesh with the source element highlighted. Log
    color scale, values below VMIN (incl. exactly zero) are shown white.
    """
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_under("white")
    coll = PolyCollection(xe, array=field / field.max(), cmap=cmap,
                          norm=LogNorm(vmin=VMIN, vmax=1.),
                          edgecolors="0.5", linewidths=0.3)
    ax.add_collection(coll)
    ax.add_collection(PolyCollection(xe[[source]], facecolors="none",
                                     edgecolors="tab:red", linewidths=1.5))
    ax.add_patch(Circle(SOURCE, R, fill=False, linestyle="--",
                        edgecolor="k", linewidth=1.))
    ax.set_xlim(0, LX)
    ax.set_ylim(0, LY)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    return coll


if __name__ == "__main__":
    with TemporaryDirectory() as tmpdir:
        meshfile = join(tmpdir, "slit.msh")
        create_mesh(meshfile)
        mesh = read_mesh(meshfile)
    coords, elnodes = mesh["coords"], mesh["elnodes"]
    xe = coords[elnodes]
    areas, centroids = element_areas_centroids(xe)
    # unit density in element closest to source location
    source = np.argmin(np.linalg.norm(centroids - SOURCE, axis=1))
    rho = np.zeros(elnodes.shape[0])
    rho[source] = 1.
    #
    fig, axs = plt.subplots(2, 1, figsize=(5.5, 7.2))
    plot_field(axs[0], xe, convolution_filter(centroids, areas, rho), source,
               "convolution filter")
    coll = plot_field(axs[1], xe, helmholtz_filter(coords, elnodes, rho),
                      source, "Helmholtz PDE filter")
    fig.colorbar(coll, ax=axs, shrink=0.8, label="normalized filter response", extend="min")
    fig.savefig(join(dirname(abspath(__file__)), "pde_filter_gaps.png"),
                bbox_inches="tight", dpi=200)
