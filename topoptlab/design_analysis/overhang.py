# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any
from functools import partial

import numpy as np
from scipy.ndimage import correlate

from topoptlab.utils import map_eltoimg, map_eltovoxel, map_imgtoel, map_voxeltoel

def baseplate_kernel(baseplate: str,
                     ndim: int) -> np.ndarray:
    """
    Return a (3,3) or (3,3,3) kernel with 1s on the stencil positions that
    face the baseplate. Convolving the density image with this kernel gives,
    for each element, the summed density of its immediate supporting neighbors.

    In 2D the entire face (row or column) of the 3x3 kernel that is adjacent
    to the baseplate is set to 1.

    In 3D only the center element of the support face plus the four
    cross-shaped arm elements are set to 1 (a plus pattern), matching the
    three-neighbor supporting region used in the Langelaar AM filter.

    Parameters
    ----------
    baseplate : str
        Build-plate orientation. In 2D: ``"N"``, ``"S"``, ``"E"``, ``"W"``.
        In 3D: ``"B"`` (bottom z), ``"U"`` (top z), ``"N"``, ``"S"``,
        ``"E"``, ``"W"``.
    ndim : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    kernel : np.ndarray, shape (3, 3) or (3, 3, 3)
        Binary kernel aligned with the baseplate direction.
    """
    baseplate = baseplate.upper()
    kernel = np.zeros(ndim * [3])

    if ndim == 2:
        table = {"N": (0, slice(None)),
                 "S": (-1, slice(None)),
                 "E": (slice(None), -1),
                 "W": (slice(None), 0)}
        kernel[table[baseplate]] = 1.
    elif ndim == 3:
        # normal_axis, plane_index
        table = {"B": (0, 0),
                 "U": (0, -1),
                 "N": (1, 0),
                 "S": (1, -1),
                 "W": (2, 0),
                 "E": (2, -1),}
        #
        normal_axis, plane_index = table[baseplate]
        center = [1, 1, 1]
        center[normal_axis] = plane_index
        # center point of support face
        kernel[tuple(center)] = 1.
        # cross arms in the two tangential directions
        tangential_axes = [ax for ax in range(3) if ax != normal_axis]
        for ax in tangential_axes:
            for pos in (0, 2):
                idx = center.copy()
                idx[ax] = pos
                kernel[tuple(idx)] = 1.
    else:
        raise ValueError("ndim must be 2 or 3")

    return kernel

def baseplate_slice(baseplate: str,
                    ndim: int) -> tuple:
    """
    Return an index tuple that selects the baseplate layer from the
    image or voxel array.

    The returned tuple can be used directly to index a (nely, nelx) or
    (nelz, nely, nelx) array. Elements in this layer are attached to the
    build platform and are considered trivially supported.

    Parameters
    ----------
    baseplate : str
        Build-plate orientation. In 2D: ``"N"``, ``"S"``, ``"E"``, ``"W"``.
        In 3D: ``"B"`` (bottom z), ``"U"`` (top z), ``"N"``, ``"S"``,
        ``"E"``, ``"W"``.
    ndim : int
        Number of spatial dimensions (2 or 3).

    Returns
    -------
    slicing : tuple
        Index tuple suitable for selecting the baseplate layer, e.g.
        ``(-1, slice(None))`` for ``"S"`` in 2D.
    """
    if ndim == 2:
        table = {"N": (0, slice(None)),
                 "S": (-1, slice(None)),
                 "E": (slice(None), -1),
                 "W": (slice(None), 0),}
    elif ndim == 3:
        table = {"B": (0, slice(None), slice(None)), 
                 "U": (-1, slice(None), slice(None)),
                 "N": (slice(None), 0, slice(None)),
                 "S": (slice(None), -1, slice(None)),
                 "E": (slice(None), slice(None), -1),
                 "W": (slice(None), slice(None), 0)}
    else:
        raise ValueError("ndim must be 2 or 3")

    return table[baseplate]

def support_indicator(nelx: int,
                      nely: int,
                      nelz: int,
                      xPhys: np.ndarray, 
                      baseplate: str,
                      to_img: bool = False, 
                      **kwargs: Any) -> np.ndarray:
    """
    Identify unsupported solid elements for an additive-manufacturing
    overhang constraint. It is taken from Eq. 1 and its related contexts in

    Zhao, Dengyang, Ming Li, and Yusheng Liu. "Self-supporting topology 
    optimization for additive manufacturing." arXiv preprint arXiv:1708.07364 (2017).

    The function computes, for each element, the amount of material in its
    supporting stencil using a convolution with the kernel associated with the
    chosen baseplate direction. An element is marked as unsupported if it has
    nonzero density but no material in its supporting region.

    Parameters
    ----------
    nelx : int
        Number of elements in x-direction.
    nely : int
        Number of elements in y-direction.
    nelz : int or None
        Number of elements in z-direction. If None, a 2D grid is assumed.
    xPhys : np.ndarray
        Physical element densities of shape ``(nel, 1)`` or compatible with
        the element-to-image/voxel mapping.
    baseplate : str
        Baseplate/build direction. In 2D: ``"N"``, ``"S"``, ``"E"``, ``"W"``.
        In 3D: ``"B"``, ``"U"``, ``"N"``, ``"S"``, ``"E"``, ``"W"``.
    to_img : bool, optional
        If False (default), the result is returned as an element array of
        shape ``(nel, k)``. If True, it is returned in image/voxel layout,
        shape ``(nely, nelx, k)`` in 2D or ``(nelz, nely, nelx, k)`` in 3D.

    Returns
    -------
    indicator : np.ndarray
        Boolean array with the same element layout as ``xPhys``. ``True``
        indicates a solid element with no support in the prescribed support
        stencil; ``False`` indicates either void material or supported material.
    """
    #
    if nelz is None:
        ndim = 2
        mapping = partial(map_eltoimg,
                          nelx=nelx,
                          nely=nely) 
        invmapping = partial(map_imgtoel,
                             nelx=nelx,
                             nely=nely)
    else:
        ndim = 3
        mapping = partial(map_eltovoxel,
                          nelx=nelx,
                          nely=nely,
                          nelz=nelz)
        invmapping = partial(map_voxeltoel,
                             nelx=nelx,
                             nely=nely,
                             nelz=nelz)
    #
    kernel = baseplate_kernel(baseplate=baseplate,
                              ndim=ndim)
    slicing = baseplate_slice(baseplate=baseplate,
                              ndim=ndim)
    #
    xPhys_map = mapping(xPhys)
    unsupported = np.isclose(correlate(xPhys_map,
                                       weights=kernel,
                                       axes=(0,1,2)[:ndim],
                                       mode="constant",
                                       cval=0.),
                           0.) & ~np.isclose(xPhys_map, 0.)
    # exclude baselayer
    unsupported[slicing] = False
    if not to_img:
        unsupported = invmapping(unsupported)
    return unsupported
