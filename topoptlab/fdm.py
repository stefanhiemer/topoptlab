# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def laplacian_2d(f: np.ndarray,dx: float) -> np.ndarray:
    """
    Laplacian operator in 2D on rectangular even spaced grid with periodic 
    boundary conditions.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_x,ngrid_y)
        function values.
    dx : float 
        lattice spacing

    Returns
    -------
    laplacian : np.ndarray shape (ngrid_x,ngrid_y)
        laplacian values.
    """
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +\
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f) / dx**2
        
    
def laplacian_3d(f: np.ndarray,dx: float) -> np.ndarray:
    """
    Laplacian operator in 3D on rectangular even spaced grid with periodic 
    boundary conditions.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_x,ngrid_y,ngrid_z)
        function values.
    dx : float 
        lattice spacing

    Returns
    -------
    laplacian : np.ndarray shape (ngrid_x,ngrid_y,ngrid_z)
        laplacian values.
    """
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +\
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +\
            np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6 * f) / dx**2