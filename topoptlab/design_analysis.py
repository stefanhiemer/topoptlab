# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union

import numpy as np
from scipy.ndimage import grey_opening, grey_closing

from topoptlab.geometries import sphere, ball
from topoptlab.utils import map_eltoimg, map_eltovoxel

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
