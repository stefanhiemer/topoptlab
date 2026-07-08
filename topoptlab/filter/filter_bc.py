# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
from typing import Any, Callable, List, Union

import numpy as np
from scipy.ndimage import binary_dilation

from topoptlab.geometries import bounding_box, sphere, ball
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel


def _face_node_indices(side: str,
                       nelx: int, nely: int,
                       nelz: Union[None, int]) -> np.ndarray:
    """
    Return 0-based node indices on the named face of a structured mesh.

    Node numbering (column-major / Fortran-like):
      2-D: node(ix, iy) = ix*(nely+1) + iy
      3-D: node(ix, iy, iz) = iz*(nelx*(nely+1)) + ix*(nely+1) + iy
    """
    if nelz is None:
        ny1 = nely + 1
        nx1 = nelx + 1
        if side == "l":
            return np.arange(ny1)
        elif side == "r":
            return np.arange(nelx * ny1, (nelx + 1) * ny1)
        elif side == "t":
            return np.arange(0, nx1 * ny1, ny1)
        elif side == "b":
            return np.arange(nely, nx1 * ny1, ny1)
        else:
            return np.array([], dtype=np.int32)
    else:
        ny1 = nely + 1
        nx1 = nelx + 1
        nz1 = nelz + 1
        n_per_z = nx1 * ny1
        iz = np.arange(nz1)
        if side == "l":
            ix = 0
            iy = np.arange(ny1)
            return (iz[:, None] * n_per_z + ix * ny1 + iy[None, :]).ravel()
        elif side == "r":
            ix = nelx
            iy = np.arange(ny1)
            return (iz[:, None] * n_per_z + ix * ny1 + iy[None, :]).ravel()
        elif side == "t":
            iy = 0
            ix = np.arange(nx1)
            return (iz[:, None] * n_per_z + ix[None, :] * ny1 + iy).ravel()
        elif side == "b":
            iy = nely
            ix = np.arange(nx1)
            return (iz[:, None] * n_per_z + ix[None, :] * ny1 + iy).ravel()
        elif side == "f":
            return np.arange(n_per_z)
        elif side == "k":
            return np.arange(nelz * n_per_z, (nelz + 1) * n_per_z)
        else:
            return np.array([], dtype=np.int32)


def bc_in_bdbox(fixed: np.ndarray,
                f: np.ndarray,
                vectorfield: bool,
                ndim: int,
                nelx: int,
                nely: int,
                nelz: Union[None, int],
                bd_box: np.ndarray,
                mirror_sides: List[str],
                create_edofMat: Callable,
                mapping,
                invmap,
                structure: np.ndarray) -> np.ndarray:
    """
    Return a boolean element mask of BC-adjacent elements within the boundary box.

    Identifies all elements whose corner nodes carry a Dirichlet (``fixed``) or
    Neumann (``f``) BC, intersects with ``bd_box``, then dilates by ``structure``
    so that a band of elements around each BC node is flagged.

    Mirror-symmetry DOFs in ``fixed`` are excluded before the masking: a mirror
    face with outward normal in direction d only constrains component d of the
    displacement (vectorfield) or the scalar DOF on that face.  These DOFs
    should not mark the adjacent elements as active.

    Parameters
    ----------
    fixed : np.ndarray
        DOF indices with Dirichlet BCs.
    f : np.ndarray
        Force/flux array of shape ``(ndof,)`` or ``(ndof, n_load_cases)``.
    vectorfield : bool
        If True, DOF indices are divided by ``ndim`` to obtain node indices
        (needed for vector fields such as 2-D/3-D linear elasticity).
    ndim : int
        Spatial dimension (2 or 3); used when ``vectorfield=True``.
    nelx, nely : int
        Number of elements in x and y.
    nelz : int or None
        Number of elements in z; None for 2-D.
    bd_box : np.ndarray of bool, shape (nel,)
        Boundary-box mask from ``bounding_box``; only elements inside this
        mask are considered.
    mirror_sides : list of str
        Faces that carry mirror/symmetry BCs.  DOFs constrained solely by
        symmetry are stripped from ``fixed`` before masking.
    mapping : callable
        Maps a 1-D element array to an nD image (partial of map_eltoimg /
        map_eltovoxel with mesh dimensions already bound).
    invmap : callable
        Inverse of ``mapping``.
    structure : np.ndarray of bool
        Structuring element for ``binary_dilation`` (disk / ball of radius rmin).

    Returns
    -------
    mask_bc : np.ndarray of bool, shape (nel,)
        True for elements that contain a real (non-symmetry) BC node and lie
        within the boundary box.

    Notes
    -----
    Refactoring opportunity: the full ``edofMat`` (nel × nodes_per_element) is
    built here only to check which elements contain a BC node.  A cheaper
    alternative would be to invert the node-to-element map — i.e. for each BC
    node index, directly look up which elements share that node — avoiding the
    full ``(nel, nnode)`` allocation.  This matters for large 3-D meshes where
    ``edofMat`` can be several hundred MB.
    """
    # Exclude DOFs that arise from mirror symmetry, not real supports.
    # For a vectorfield: face normal in direction d → only DOFs with
    # (dof % ndim == d) on that face are symmetry DOFs.
    if vectorfield and mirror_sides:
        face_to_component = {"l": 0, "r": 0, "t": 1, "b": 1, "f": 2, "k": 2}
        sym_dofs = []
        for side in mirror_sides:
            comp = face_to_component.get(side)
            if comp is None or comp >= ndim:
                continue
            nodes = _face_node_indices(side, nelx, nely, nelz)
            sym_dofs.append(nodes * ndim + comp)
        if sym_dofs:
            sym_dofs = np.unique(np.concatenate(sym_dofs))
            fixed = fixed[~np.isin(fixed, sym_dofs)]

    #
    bc_ndinds = np.hstack((fixed, f.nonzero()[0]))
    #
    if vectorfield:
        #
        bc_ndinds = np.unique(np.floor(bc_ndinds/ndim).astype(np.int32))
    # 
    edofMat= create_edofMat(nelx=nelx, nely=nely, nelz=nelz,
                            nnode_dof=1)[0]
    mask_bc = np.isin(edofMat,bc_ndinds).any(axis=1)
    #mask_bc = invmap(binary_dilation(mapping(mask_bc & bd_box), 
    #                 structure=structure))
    return mask_bc

def create_filter_bc(nelx: int,
                     nely: int,
                     rmin: float,
                     fixed: np.ndarray,
                     f: np.ndarray,
                     mirror_sides: Union[List[str]] = [],
                     el_flags: Union[None, np.ndarray] = None,
                     nelz: Union[None, int] = None, 
                     vectorfield: bool = False, 
                     wrapping: bool = False,
                     **kwargs: Any) -> np.ndarray:
    """
    Create element flags to apply boundary conditions similar to 

       Clausen, Anders, and Erik Andreassen. "On filter boundary 
       conditions in topology optimization." Structural and 
       Multidisciplinary Optimization 56.5 (2017): 1147-1155.

    This is needed for consistent optimization at the boundaries of 
    the design but also for special filters e. g. coating. This is 
    done by building or augmenting an el_flags array.

    Applied in order (later steps override earlier ones):

    1. **Boundary strip (flag=3)**: the outermost ``ceil(rmin)`` element layers
       on each non-mirror face of the bounding box are set to non-design. The
       band is at least ``ceil(rmin)`` elements wide so the density filter
       cannot smear through it.
    2. **BC elements (flag=2)**: boundary elements whose corner nodes carry
       Dirichlet (``fixed``) or Neumann (``f``) BCs are set to active (solid),
       overriding the flag=3 from step 1.
    3. **Wrap passive regions (flag=3)**: all elements within a dilation of
       radius ``ceil(rmin)`` around every passive element (flag=1) that are
       currently free (flag=0) are marked non-design.  This prevents free
       elements from being closer than rmin to a prescribed region.

    Parameters
    ----------
    nelx : int
        number of elements in x.
    nely : int
        number of elements in y.
    rmin : float
        filter radius in element widths; controls the minimum band width.
    mirror_sides : list of str
        sides with mirror/symmetry BCs — no flag=3 strip is added on these.
        Valid values: ``"l"``, ``"r"``, ``"t"``, ``"b"`` (and ``"f"``,
        ``"k"`` for 3D).
    el_flags : np.ndarray of int or None
        existing flags to augment.  If None, a zero array is created.
    fixed : np.ndarray or None
        DOF indices with Dirichlet BCs, as returned by ``bcs()``.
    f : np.ndarray or None
        force array of shape ``(ndof,)`` or ``(ndof, n_load_cases)`` with
        Neumann BCs, as returned by ``bcs()``.
    nelz : int or None
        number of elements in z (3D); None for 2D.

    Returns
    -------
    el_flags : np.ndarray of int, shape (n,)
        updated element flags of dtype np.int8.
    """
    #
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
        mapping = partial(map_eltoimg, nelx=nelx, nely=nely)
        invmap  = partial(map_imgtoel, nelx=nelx, nely=nely)
        struct  = sphere
    else:
        ndim = 3
        create_edofMat = create_edofMat3d
        mapping = partial(map_eltovoxel, nelx=nelx, nely=nely, nelz=nelz)
        invmap  = partial(map_voxeltoel, nelx=nelx, nely=nely, nelz=nelz)
        struct  = ball
    #
    l = 1+int(2*rmin)
    if nelz is None:
        center = np.array([l // 2, l // 2])
        structure = struct(nelx=l, nely=l, center=center,
                           radius=rmin, fill_value=1).reshape(l, l).astype(bool)
    else:
        center = np.array([l // 2, l // 2, l // 2])
        structure = struct(nelx=l, nely=l, nelz=l, center=center,
                           radius=rmin, fill_value=1).reshape(l, l, l).astype(bool)
    #
    n_layers = np.maximum(1, np.ceil(rmin)).astype(np.int32)
    n = np.prod([nelx,nely,nelz][:ndim])
    #
    if el_flags is None:
        el_flags = np.zeros(n, dtype=np.int8)
    else:
        el_flags = el_flags.copy()
    # get bounding box (flag=3)
    all_faces = {"l", "r", "t", "b", "f", "k"}
    active_faces = list(all_faces - set(mirror_sides))
    # get bounding box
    bd_box = bounding_box(nelx=nelx, nely=nely, nelz=nelz,
                          faces=active_faces,
                          thickness=n_layers,
                          fill_value=True)
    # find elements with FE boundary conditions applied
    # that lie in bounding box.
    mask_bc = bc_in_bdbox(fixed=fixed,
                          f=f,
                          vectorfield=vectorfield,
                          ndim=ndim,
                          nelx=nelx,
                          nely=nely,
                          nelz=nelz,
                          bd_box=bd_box,
                          mirror_sides=mirror_sides,
                          create_edofMat=create_edofMat,
                          mapping=mapping,
                          invmap=invmap,
                          structure=structure)
    # wrap passive regions with flag=3
    wrapped = mapping(el_flags == 1)
    if wrapped.any():
        #
        wrapped[binary_dilation(wrapped, 
                                structure=structure) &\
                (wrapped == 0)] = True
    wrapped = invmap(wrapped)
    wrapped[el_flags != 0] = False
    # set elements in bounding box at FE bc to active
    el_flags[mask_bc  & (el_flags == 0)] = 2
    # 
    el_flags[(bd_box | wrapped) & (el_flags == 0)] = 3
    return el_flags
