# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.poisson_2d import lk_poisson_2d ,lk_poisson_aniso_2d
from topoptlab.elements.mass_scalar_2d import lm_mass_2d

def lk_screened_poisson_2d(k: float = 1.,
                           l: np.ndarray = np.array([1.,1.]), 
                           g: np.ndarray = np.array([0.]),
                           t: float = 1.,
                           **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for screened 2D Poisson with bilinear
    quadrilateral elements.

    Parameters
    ----------
    k : float
        analogous to heat conductivity. k is the squared filter radius if used 
        for the 'Helmholtz' filter.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element.

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    return lk_poisson_2d(k=k, l=l, g=g, t=t) + lm_mass_2d(p=1., l=l, t=t)

def lk_screened_poisson_aniso_2d(k: np.ndarray,
                                 l: np.ndarray = np.array([1.,1.]), 
                                 g: np.ndarray = np.array([0.]),
                                 t: float = 1.,
                                 **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for anisotropic 2D screened Poisson with 
    bilinear quadrilateral elements.

    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        analogous to anisotropic heat conductivity. If isotropic k would be 
        [[k,0],[0,k]] and k is the squared filter radius if used for the 
        'Helmholtz' filter.
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    return lk_poisson_aniso_2d(k=k, l=l, g=g, t=t) + lm_mass_2d(p=1., l=l, t=t)
