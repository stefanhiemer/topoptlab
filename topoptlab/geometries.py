# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union,List

import numpy as np

def sphere(nelx: int, nely: int, center: np.ndarray, 
           radius: float, 
           fill_value: int =1) -> np.ndarray:
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

def bounding_rectangle(nelx: int, nely: int, 
                       faces: List = ["b","t","r","l"]) -> np.ndarray:
    """
    Create element flags for a bounding box of one element thickness. It is
    possible to draw only specified faces of the bounding box.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    faces : list of str
        which faces of bounding box are supposed to be drawn. Possible
        values are "b" for bottom, "t" for top, "l" for left and "r" for right.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    # collect indices
    indices = []
    # append corner indices
    if "t" in faces or "l" in faces:
        indices.append(0)
    if "t" in faces or "r" in faces:
        indices.append((nelx-1)*nely)
    if "b" in faces or "l" in faces:
        indices.append(nely - 1)
    if "b" in faces or "r" in faces:
        indices.append(nelx*nely-1)
    # append faces without corner indices
    if "t" in faces:
        indices.append(np.arange(nely,(nelx-1)*nely,nely))
    if "b" in faces:
        indices.append(np.arange(nely-1,nelx*nely,nely))
    if "l" in faces:
        indices.append(np.arange(1,nely-1))
    if "r" in faces:
        indices.append(np.arange((nelx-1)*nely + 1,nelx*nely-1))
    #
    indices = np.hstack(indices)
    el_flags = np.zeros(nelx*nely,dtype=int)
    # set to active
    el_flags[indices] = 2
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
