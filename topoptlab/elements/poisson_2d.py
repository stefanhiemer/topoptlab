# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

def lk_poisson_2d(k: float = 1.,
                  l: np.ndarray = np.array([1.,1.]), 
                  g: np.ndarray = np.array([0.]),
                  t: float = 1.) -> np.ndarray:
    """
    Create element stiffness matrix for 2D Poisson with bilinear
    quadrilateral elements.

    Parameters
    ----------
    k : float
        heat conductivity.
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
    return t*k*np.array([[l[0]/(3*l[1]) - np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) + np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2)],
                         [l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) + np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) - np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2)],
                         [-l[0]/(6*l[1]) + np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) - np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2)],
                         [-l[0]/(3*l[1]) + l[1]/(6*l[0]*np.cos(g[0])**2), -l[0]/(6*l[1]) - np.tan(g[0])/2 - l[1]/(6*l[0]*np.cos(g[0])**2), l[0]/(6*l[1]) - l[1]/(3*l[0]*np.cos(g[0])**2), l[0]/(3*l[1]) + np.tan(g[0])/2 + l[1]/(3*l[0]*np.cos(g[0])**2)]])

def lk_poisson_aniso_2d(k: np.ndarray,
                        l: np.ndarray = np.array([1.,1.]), 
                        g: np.ndarray = np.array([0.]),
                        t: float = 1.) -> np.ndarray:
    """
    Create element stiffness matrix for anisotropic 2D Poisson with bilinear
    quadrilateral elements.

    Parameters
    ----------
    k : np.ndarray, shape (2,2)
        anisotropic heat conductivity. If isotropic k would be [[k,0],[0,k]]
    l : np.ndarray (2)
        side length of element
    t : float
        thickness of element

    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.

    """
    return t*np.array([[k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(3*l[1]) - k[1,1]*np.tan(g[0])/2 + k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           -k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           -k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) + k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(3*l[1]) + k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0])],
          [-k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           (4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(-k[0,1] - k[1,0] + 2*k[1,1]*np.tan(g[0])) + 4*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           (-4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(k[0,1] - k[1,0]) + 2*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           -k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) - k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0])],
          [-k[0,0]*l[1]/(6*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) + k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           (-4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(-k[0,1] + k[1,0]) + 2*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           (4*k[1,1]*l[0]**2 + 3*l[0]*l[1]*(k[0,1] + k[1,0] - 2*k[1,1]*np.tan(g[0])) + 4*l[1]**2*(k[0,0] - k[0,1]*np.tan(g[0]) - k[1,0]*np.tan(g[0]) + k[1,1]*np.tan(g[0])**2))/(12*l[0]*l[1]),
           -k[0,0]*l[1]/(3*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0])],
          [k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(3*l[1]) + k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           -k[0,0]*l[1]/(6*l[0]) + k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(6*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(6*l[0]) - k[1,1]*l[0]/(6*l[1]) - k[1,1]*np.tan(g[0])/2 - k[1,1]*l[1]*np.tan(g[0])**2/(6*l[0]),
           -k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 + k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,0]/4 + k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(6*l[1]) - k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0]),
           k[0,0]*l[1]/(3*l[0]) - k[0,1]/4 - k[0,1]*l[1]*np.tan(g[0])/(3*l[0]) - k[1,0]/4 - k[1,0]*l[1]*np.tan(g[0])/(3*l[0]) + k[1,1]*l[0]/(3*l[1]) + k[1,1]*np.tan(g[0])/2 + k[1,1]*l[1]*np.tan(g[0])**2/(3*l[0])]])
