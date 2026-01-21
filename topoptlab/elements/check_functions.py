# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple,Any,Union 

import numpy as np

def check_inputs(*coords: Union[float, np.ndarray,None], 
                 ndim: int,
                 nnodes: int,
                 xe: Union[None,np.ndarray] = None,
                 all_elems: bool = False,
                 **kwargs: Any) -> Tuple:
    """
    Check coordinates and provided element node information for consistency.
    If necessary transform inputs to make them consistent. The function 
    enforces the following consistency rules:
        
        - all coordinates must be either scalars or one-dimensional arrays of 
          equal length.
        - returns up to three reference coordinates and are padded with None 
          for compatibility across dimensions.
        - element-local data may correspond to a single element, one element 
          per reference coordinate, or one element per group of support points.
        - if `all_elems` is False, incompatible combinations of element count 
          and reference coordinate count raise an error.
        - if `all_elems` is True, all reference coordinates are evaluated for 
          all elements by repeating and tiling inputs as needed.

    Parameters
    ----------
    *coords : float or np.ndarray
        Reference coordinates (xi, eta, zeta) in the reference domain.
        Each coordinate must be either a scalar or a 1D array of shape (ncoords).
        The number of coordinates must equal `ndim`.
    ndim : int
        Spatial dimension of the element.
    nnodes : int
        Number of nodes per element.
    xe : np.ndarray
        coordinates of element nodes shape (nels,nnodes,ndim). nels must be either 1,
        ncoords/nnodes or the equal to ncoords. The two exceptions are if
        ncoords = 1 or all_elems is True. 
    all_elems : bool
        If True, coordinates are evaluated for all elements.

    Returns
    -------
    if xe is None
    ncoords : int
        Number of coordinates.

    if xe is not None
    xe : np.ndarray
        Coordinates of element nodes of shape (n, nnodes, ndim).
    *coords : tuple
        coordinates, each of shape (n) or None to achieve len(coords)=3.
    """
    # discard empty coordinates
    coords = coords[:len(coords)-sum(c is None for c in coords)]
    #
    if ndim > 3:
        raise ValueError("ndim must be <= 3.")
    # check reference coordinates
    if len(coords) != ndim:
        raise ValueError(f"expected {ndim} coordinates, got {len(coords)}")
    #
    if all(isinstance(c, np.ndarray) for c in coords):
        if any(c.ndim != 1 for c in coords):
            raise ValueError("all coordinates must be 1D np.ndarrays.")
        lengths = [c.shape[0] for c in coords]
        if len(set(lengths)) != 1:
            raise ValueError("all coordinates must have the same shape.")
        ncoords = lengths[0]

    elif all(isinstance(c, (int, float)) for c in coords):
        ncoords = 1

    else:
        raise ValueError("coordinate datatypes are inconsistent.")
    # 
    if xe is None:
        return ncoords
    #
    xe = np.asarray(xe)
    if xe.ndim==2:
        xe = xe[None,:,:]
    #
    if xe.shape[-2:] != (nnodes, ndim):
        raise ValueError(f"xe must have shape (nels, {nnodes}, {ndim}) or ", 
                         f"({nnodes}, {ndim}).")
    # check nels and ncoords for compatibility
    nels = xe.shape[0]
    if not all_elems and all([nels != ncoords,nnodes*nels != ncoords,
                              nels != 1,ncoords!=1]):
        raise ValueError("shapes of nels and ncoords incompatible.")
    # expand coordinates if needed and add None returns for dimensional 
    # consistency
    if all_elems:
        coords = tuple(np.tile(c, nels) if ncoords > 1 else np.full(nels, c)
                       for c in coords)
        xe = np.repeat(xe,repeats=ncoords,axis=0)
    elif nnodes*nels == ncoords:
        xe = np.repeat(xe,repeats=nnodes,axis=0)
    # add None for dimensional consistency
    coords = coords + tuple(None for i in range(3-len(coords)))
    return xe,*coords