# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,List, Union

import numpy as np
from scipy.ndimage import binary_dilation

from topoptlab.geometries import bounding_box, sphere, ball
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

def create_filter_bc(nelx: int,
                     nely: int,
                     rmin: float,
                     fixed: np.ndarray,
                     f: np.ndarray,
                     mirror_sides: Union[None, List[str]] = None,
                     el_flags: Union[None, np.ndarray] = None,
                     nelz: Union[None, int] = None, 
                     vectorfield: bool = False, 
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
    mirror_sides : list of str or None
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
        updated element flags.
    """
    #
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
        mapping = map_eltoimg
        invmap = map_imgtoel
        struct = sphere 
    else:
        ndim = 3
        create_edofMat = create_edofMat3d
        mapping = map_eltovoxel
        invmap = map_voxeltoel 
        struct = ball
    #
    l = 1+int(2*rmin)
    structure = struct(nelx=l,nely=l,nelz=l,
                       radius=rmin, 
                       fill_value=1.)
    #
    if mirror_sides is None:
        mirror_sides = []
    #
    n_layers = np.maximum(1, np.ceil(rmin)).astype(np.int32)
    n = np.prod([nelx,nely,nelz][:ndim])
    #
    if el_flags is None:
        el_flags = np.zeros(n, dtype=int)
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
    # that lie in bounding box
    bc_ndinds = np.hstack((fixed, f.nonzero()[0]))
    if vectorfield:
        bc_ndinds = np.unique(np.floor(bc_ndinds/ndim).astype(np.int32))
    # 
    edofMat = create_edofMat(nelx=nelx, nely=nely, nelz=nelz,
                             nnode_dof=1)
    mask_bc = np.isin(edofMat,bc_ndinds).any(axis=1)
    mask_bc = invmap(binary_dilation(mapping(mask_bc & bd_box), 
                     structure=structure))
    # wrap passive regions with flag=3
    wrapped = mapping(el_flags == 1)
    if wrapped.any():
        #
        wrapped[binary_dilation(wrapped, 
                                structure=structure) &\
                (wrapped == 0)] = True
    wrapped = invmap(wrapped)
    wrapped[el_flags != 0] = False
    # 
    el_flags[(bd_box | wrapped) & el_flags == 0] = 3
    # set elements in bounding box at FE bc to active
    el_flags[(mask_bc & bd_box) & el_flags == 3] = 2
    return el_flags
