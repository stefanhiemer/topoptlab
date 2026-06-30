# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union
from functools import partial

import numpy as np
from scipy.ndimage import grey_opening, grey_closing
from scipy.interpolate import CubicHermiteSpline

from topoptlab.geometries import sphere, ball
from topoptlab.utils import map_eltoimg, map_eltovoxel, map_imgtoel, map_voxeltoel

def gray_indicator(x: np.ndarray) -> np.ndarray:
    """
    Gray level indicator to measure discreteness of the designs as in Eq. 41 
    in 
    
    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.
    
    For a design with all densities at 1/2 returns 1, if all densities either
    1 or 0 it returns 0.

    Parameters
    ----------
    x : np.ndarray shape (n) or shape (n,nmats)
        design variables. 

    Returns
    -------
    indicator : np.ndarray of shape () or shape (nmats)
        intermediate density indicator.

    """
    return 4*(x*(1-x)).mean(axis=0)

def level_indicator(x: np.ndarray, 
                    x_i: [None,np.ndarray] = None, 
                    nlevels: Union[None,int] = None) -> np.ndarray:
    """
    Level indicator to measure discreteness of multi-level designs. Generalizes
    gray_indicator to the case where x is expected to take more than two
    discrete values x_0, x_1, ..., x_N. The indicator is constructed as a
    piecewise cubic Hermite spline with peaks of 1 halfway between adjacent
    levels and zeros at the levels themselves. Returns 0 when all densities
    coincide with a discrete level and approaches 1 when all densities sit
    exactly halfway between two adjacent levels.

    Supply either x_i or nlevels, not both.

    Parameters
    ----------
    x : np.ndarray of shape (n,) or (n, nmats)
        design variables.
    x_i : np.ndarray of shape (nlevels,) or None
        explicit discrete target levels, e.g. np.array([0., 0.5, 1.]).
    nlevels : int or None
        number of equally spaced levels in [0, 1].

    Returns
    -------
    indicator : np.ndarray of shape () or (nmats,)
        intermediate density indicator in [0, 1].

    """
    if x_i is None and isinstance(nlevels,int):
        herm = CubicHermiteSpline(x=np.linspace(0,1/nlevels,3),
                                  y=np.array([0,1,0]),
                                  dydx=np.zeros(3),
                                  extrapolate="periodic")
    elif isinstance(x_i,np.ndarray) and len(x_i.shape)==1:
        # midpoints
        x_mid = (x_i[1:]-x_i[:-1])/2
        x_knots = np.column_stack((x_i, np.append(x_mid,np.zeros(1)))).flatten()[:-1]
        #
        herm = CubicHermiteSpline(x=x_knots,
                                  y=np.tile([0,1],x_i.shape)[:-1],
                                  dydx=np.zeros(2*x_i.shape[0]-1),
                                  extrapolate=None)
    else:
        raise NotImplementedError("Input inconsistent.")
    return herm(x).mean(axis=0)

def lengthscale_violations(x: np.ndarray,
                           r: float,
                           nelx: int, nely: int,
                           nelz: Union[None,int] = None
                           ) -> [np.ndarray,np.ndarray]:
    """
    Visualize length scale violations as suggested in chapter 2.6 of 
    
    Sigmund, Ole. "On benchmarking and good scientific practise in topology 
    optimization." Structural and Multidisciplinary Optimization 65.11 (2022): 
    315.
    
    Length scale violations are determined by creating a sphere of radius r 
    and performing morphological closing and opening operations. For length 
    scale violations in the void we perform 
    
    close(x) - x
    
    while for length scale violation in solid material we do
    
    x - open(x)

    Parameters
    ----------
    x : np.ndarray shape (n)
        design variables.
    r : float
        radius of sphere that determines length scale.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None, optional
        number of elements in z direction. The default is None.

    Returns
    -------
    solidviolation : np.ndarray, shape (nely,nelx) or shape (nelz,nely,nelx)
        lengthscale violations in solid phase.
    voidviolation : np.ndarray, shape (nely,nelx) or shape (nelz,nely,nelx)
        lengthscale violations in void phase.

    """
    
    #
    r = int(r)
    l = 1+int(2*r)
    #
    if nelz is None:
        x = map_eltoimg(quant=x, nelx=nelx, nely=nely)
        structure = sphere(nelx=l, nely=l,
                           center=(np.median([0,l-1]),
                                   np.median([0,l-1])),
                           radius=r, fill_value=1.)
        structure = map_eltoimg(quant=structure, nelx=l, nely=l)
    else:
        x = map_eltovoxel(quant=x, nelx=nelx, nely=nely, nelz=nelz)
        structure = ball(nelx=l, nely=l, nelz=l,
                           center=(np.median([0,l-1]),
                                   np.median([0,l-1]),
                                   np.median([0,l-1])),
                           radius=r, fill_value=1.)
        structure = map_eltovoxel(quant=structure, nelx=l, nely=l, nelz=l)
    #
    solidviolation = x-grey_opening(x,size=structure.shape,
                                    structure=structure,
                                    mode="nearest",cval=0.)
    voidviolation = grey_closing(x,size=structure.shape,
                                 structure=structure,
                                 mode="nearest",cval=0.)-x
    return solidviolation, voidviolation

def baseplate_kernel(baseplate: str, 
                     ndim: int) -> Tuple:
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

def support_indicator(nelx: int,
                      nely: int,
                      nelz: int,
                      xPhys: np.ndarray, 
                      baseplate: str) -> np.ndarray:
    
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
    #
    indicator = invmapping(convolve(mapping(xPhys),
                                    weights=kernel, 
                                    axes=(0,1,2)[:ndim],
                                    mode="constant",
                                    cval=0.)) / self.hs
    return np.isclose(indicator, 0.) & ~np.isclose(xPhys,0.)