# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.poisson_3d import lk_poisson_3d,lk_poisson_aniso_3d
from topoptlab.elements.mass_scalar_3d import lm_mass_3d 

def lk_screened_poisson_3d(k: int = 1,
                           l: np.ndarray = np.array([1.,1.,1.]), 
                           g: np.ndarray = np.array([0.,0.]),
                           **kwargs: Any) -> np.ndarray:
    """
    Create matrix for 3D screened Poisson equation with trilinear hexahedral 
    elements.
    
    Parameters
    ----------
    k : float
        analogous to heat conductivity. k is the squared filter radius if used 
        for the 'Helmholtz' filter.
        
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    return lk_poisson_3d(k=k, l=l, g=g) + lm_mass_3d(p=1., l=l)

def lk_screened_poisson_aniso_3d(k: np.ndarray,
                                 l: np.ndarray = np.array([1.,1.,1.]), 
                                 g: np.ndarray = np.array([0.,0.]),
                                 **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for anisotropic 3D screened Poisson with 
    trilinear hexahedral elements. 
    
    Parameters
    ----------
    k : np.ndarray, shape (3,3)
        analogous to anisotropic heat conductivity. If isotropic k would be 
        [[k,0,0],[0,k,0],[0,0,k]] and k is the squared filter radius if used 
        for the 'Helmholtz' filter.
    l : np.ndarray, shape (3)
        side length of element.
    g : np.ndarray, shape  (2)
        angles of parallelogram.
        
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    return lk_poisson_aniso_3d(k=k, l=l, g=g) + lm_mass_3d(p=1., l=l)
