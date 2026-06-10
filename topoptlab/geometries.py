# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Union

import numpy as np

from topoptlab.utils import check_meshdata, elid_to_coords, nodeid_to_coords,\
                            map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

def cube_mask(coords : np.ndarray,
              low : np.ndarray, 
              upp : np.ndarray,
              **kwargs : Any) -> np.ndarray:
    """
    Take coordinate and return mask of coordinates lying between the lower 
    and upper boundaries.
    
    Parameters
    ----------
    coords : np.ndarray 
        element coordinates of shape shape (ncoords,ndim).
    low : np.ndarray
        lower boundaries of the cuboid of shape shape (ndim).
    upp : np.ndarray
        upper boundaries of the cuboid of shape shape (ndim).
    
    Returns
    -------
    mask : np.ndarray 
        mask for coordinates of shape (ncoords).
    
    """
    #
    return np.all((coords <= upp[None,:]) &\
                  (coords >= low[None,:]), 
                  axis=1)

def elids_in_mask(el: np.ndarray,
                  spatial_mask_fnc : Callable, 
                  mask_kw : Dict,
                  nelx: int, 
                  nely: int, 
                  nelz: Union[None,int] = None,
                  l: Union[float,List,np.ndarray] = [1.,1.,1.],
                  g: Union[List,np.ndarray] = [0.,0.], 
                  **kwargs: Any) -> np.ndarray:
    """
    Find element IDs within an interval of cartesian coordinates in the usual 
    regular grid. 

    Parameters
    ----------
    el : np.ndarray
        element IDs of shape (nel).
    spatial_mask_fnc : callable
        function that creates the spatial mask based on coordinates and mask_kw.
    mask_kw : 
        keywords for the spatial mask.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    l : float or list 
        side length of elements.
    g : float or list 
        angle of elements. if both angles zero, element is rectangular/cuboid.
    
    Returns
    -------
    indices : np.ndarray 
        element indices shape (n).

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    l,g = check_meshdata(l=l, 
                         g=g, 
                         ndim=ndim)
    # find coordinates of each element
    coords = elid_to_coords(el = el, 
                            nelx = nelx, 
                            nely = nely, 
                            nelz = nelz,
                            l = l,
                            g = g)
    #
    return np.nonzero(spatial_mask_fnc(coords=coords, **mask_kw))[0]

def nodeids_in_mask(node_id: np.ndarray,
                    spatial_mask_fnc : Callable, 
                    mask_kw : Dict,
                    nelx: int, 
                    nely: int, 
                    nelz: Union[None,int] = None,
                    l: Union[float,List,np.ndarray] = [1.,1.,1.],
                    g: Union[List,np.ndarray] = [0.,0.], 
                    **kwargs: Any) -> np.ndarray:
    """
    Find node IDs within an interval of cartesian coordinates in the usual 
    regular grid. 

    Parameters
    ----------
    nd_id : np.ndarray
        node IDs of shape (n_node).
    spatial_mask_fnc : callable
        function that creates the spatial mask based on coordinates and mask_kw.
    mask_kw : 
        keywords for the spatial mask.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    l : float or list 
        side length of elements.
    g : float or list 
        angle of elements. if both angles zero, element is rectangular/cuboid.
    
    Returns
    -------
    indices : np.ndarray 
        node indices shape (n).

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    l,g = check_meshdata(l=l, 
                         g=g, 
                         ndim=ndim)
    # find coordinates of each element
    coords = nodeid_to_coords(nd = node_id,
                              nelx = nelx, 
                              nely = nely, 
                              nelz = nelz,
                              l = l,
                              g = g)
    #
    return np.nonzero(spatial_mask_fnc(coords=coords, **mask_kw))[0]

def sphere(nelx: int, nely: int, center: np.ndarray, 
           radius: float, 
           fill_value: int =1, 
           **kwargs: Any) -> np.ndarray:
    """
    Create element flags for a sphere located at the specified center with the
    specified radius.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    center : np.ndarray
        coordinates of sphere center.
    radius : float
        sphere radius.
    fill_value: int
        value that is prescribed to elements within sphere.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    n = nelx*nely
    el = np.arange(n, dtype=np.int32)
    i,j = np.divmod(el,nely)
    mask = (i-center[0])**2 + (j-center[1])**2 <= radius**2
    #
    el_flags = np.zeros(n,dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def ellipse(nelx: int, nely: int,
            center: np.ndarray,
            ax_half_lengths: np.ndarray,
            fill_value: int = 1) -> np.ndarray:
    """
    Create element flags for an axis-aligned ellipse.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    center : np.ndarray
        (cx, cy) coordinates of ellipse center.
    ax_half_lengths : np.ndarray
        (a, b) ellipse semi-axes lengths.
    fill_value : int
        value assigned to elements inside the ellipse.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)
    """
    n = nelx * nely
    el = np.arange(n, dtype=np.int32)
    i, j = np.divmod(el, nely)

    # ellipse equation: ((x-cx)^2)/a^2 + ((y-cy)^2)/b^2 ≤ 1
    a, b = ax_half_lengths
    mask = ((i - center[0])**2) / a**2 + ((j - center[1])**2) / b**2 <= 1.0

    el_flags = np.zeros(n, dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def ball(nelx: int, nely: int, nelz: int, 
         center: np.ndarray, radius: float, 
         fill_value: int = 1) -> np.ndarray:
    """
    Create element flags for a ball located at the specified center with the
    specified radius.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    center : list or tuple or np.ndarray
        coordinates of sphere center.
    radius : float
        sphere radius.
    fill_value: int
        value that is prescribed to elements within ball.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely*nelz)

    """
    n = nelx*nely*nelz
    el = np.arange(n, dtype=np.int32)
    k,ij = np.divmod(el,nelx*nely)
    i,j = np.divmod(ij,nely)
    mask = (i-center[0])**2 + (j-center[1])**2 + (k-center[2])**2 <= radius**2
    #
    el_flags = np.zeros(n, dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def ellipsoid(nelx: int, nely: int, nelz: int,
            center: np.ndarray,
            ax_half_lengths: np.ndarray,
            fill_value: int = 1) -> np.ndarray:
    """
    Create element flags for an axis-aligned ellipse.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    center : np.ndarray
        (cx, cy, cz) coordinates of ellipse center.
    ax_half_lengths : np.ndarray
        (a, b,c) ellipse semi-axes lengths.
    fill_value : int
        value assigned to elements inside the ellipse.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)
    """
    n = nelx*nely*nelz
    el = np.arange(n, dtype=np.int32)
    k,ij = np.divmod(el,nelx*nely)
    i,j = np.divmod(ij,nely)
    # ellipse equation: ((x-cx)^2)/a^2 + ((y-cy)^2)/b^2 ≤ 1
    a, b = ax_half_lengths
    mask = ((i - center[0])**2) / a**2 + ((j - center[1])**2) / b**2 <= 1.0

    el_flags = np.zeros(n, dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags

def diracdelta(nelx: int, nely: int, nelz: Union[None,int] = None,
               location: Union[None,int] = None) -> np.ndarray:
    """
    Create element flags for a Dirac delta located at the specified location.
    Depending on the location and the number of elements in each direction this
    results in either a single element with flag 1 or 4/8 elements in 2/3
    dimensions.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    location : list or tuple or np.ndarray
        coordinate of Dirac delta.

    Returns
    -------
    el_flags : np.ndarray shape (nelx*nely) or shape (nelx*nely*nelz)
        element flags / densities

    """
    if location is None and nelz is None:
        location = ((nelx-1)/2,(nely-1)/2)
    elif location is None and nelz is not None:
        location = ((nelx-1)/2,(nely-1)/2,(nelz-1)/2)
    # densities
    if nelz is None:
        x = sphere(nelx=nelx,nely=nely,
                   center=location,
                   radius=1,fill_value=1.)
    else:
        x = ball(nelx=nelx,nely=nely,nelz=nelz,
                 center=location,
                 radius=1,fill_value=1.)
    return x

def bounding_box(nelx: int, nely: int,
                 faces: List[str] = ["b", "t", "r", "l"],
                 fill_value: int = 2,
                 thickness: Union[None,int] = None,
                 nelz: Union[None, int] = None) -> np.ndarray:
    """
    Create element flags for the boundary shell of the mesh in 2D or 3D.

    Element indexing (column-major / Fortran-like within each x-column):

    - 2D: ``e = ex * nely + ey``
    - 3D: ``e = (ez * nelx + ex) * nely + ey``

    Face labels for 2D: ``"l"`` (left, ex=0), ``"r"`` (right, ex=nelx-1),
    ``"t"`` (top, ey=0), ``"b"`` (bottom, ey=nely-1).
    Additional labels for 3D: ``"f"`` (front, ez=0), ``"k"`` (back, ez=nelz-1).

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    faces : list of str
        which faces to include.
    thickness : int
        number of element layers per face (default 1).
    fill_value : int
        flag value assigned to selected elements (default 2 = active).
    nelz : int or None
        number of elements in z direction; None for 2D.

    Returns
    -------
    el_flags : np.ndarray of int, shape (n,)
        element flags with ``fill_value`` set on the selected faces.

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if thickness and ndim == 2:
        mapping = map_eltoimg
        invmap = map_imgtoel
    elif thickness and dim == 3:
        mapping = map_eltovoxel
        invmap = map_voxeltoel
    #
    n = np.prod([nel,nely,nelz][:ndim])
    #
    inds = []
    if "l" in faces:
        inds += [np.arange(nely)]
    if "r" in faces:
        inds += [np.arange(nely)+nely*(nelx-1)]
    if "t" in faces:
        inds += [np.arange(0,n,nely)]
    elif "b" in faces:
        inds += [np.arange(0,n,nely)+nely-1]
    #
    if ndim == 3:
        #
        if len(inds) !=0:
            inds = np.unique(np.array(inds)) 
            #
            inds = [inds[:,None] + (np.arange(nelz)*nelx*nely)[None,:]]
        #
        if "f" in faces:
            inds += [np.arange(nelx*nely)]
        if "k" in faces:
            inds += [np.arange(nelx*nely) + nelx*nely*(nelz-1) - 1]
    if len(inds) !=0:
        inds = np.unique(np.array(inds))
    #
    if isinstance(fill_value,(int,np.in32,np.int64)):
        el_flags = np.zeros(n, dtype=np.int32)
    elif isinstance(fill_value,(float,np.float64)):
        el_flags = np.zeros(n, dtype=np.float64)
    elif isinstance(fill_value,bool):
        el_flags = np.zeros(n, dtype=bool)
    el_flags[inds] = fill_value
    #
    if thickness and el_flags.dtype == np.int32:
        el_flags = invmap(grey_dilation(mapping(el_flags),
                          size=thickness)).astype(np.int32)
    elif thickness and el_flags.dtype == np.float64:
        el_flags = invmap(grey_dilation(mapping(el_flags),
                          size=thickness))
    elif thickness and el_flags.dtype == bool:
        el_flags = invmap(binary_dilation(mapping(el_flags),
                          size=thickness))
    return el_flags

def slab(nelx: int, nely: int, center: np.ndarray, 
         widths: Union[None,List] = None, fill_value: int = 1) -> np.ndarray:
    """
    Create element flags for a slab located at the specified center with the
    specified width in each dimension.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    center : list or tuple or np.ndarray
        coordinates of slab center.
    widths : iterable of floats and None
        width in x and y direction. If one entry is None or it is width is None,
        then nelx/nely is taken as width in this direction.
    fill_value: int
        value that is prescribed to elements within sphere.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    #
    widths = [ [nelx,nely][i] if w is None else w for i,w in enumerate(widths)]
    #
    n = nelx*nely
    el = np.arange(n, dtype=np.int32)
    i,j = np.divmod(el,nely)
    #
    mask = (np.abs(i-center[0]) <= widths[0]/2 ) & (np.abs(j-center[1]) <= widths[1]/2)
    #
    el_flags = np.zeros(n,dtype=np.int32)
    el_flags[mask] = fill_value
    return el_flags
