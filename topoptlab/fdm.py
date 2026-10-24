# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any
import numpy as np

def laplacian_bc(L: np.ndarray,
                 phi: None | np.ndarray = None,
                 free: None | np.ndarray = None,
                 fixed: None | np.ndarray = None,
                 dirichlet_inds: None | np.ndarray = None,
                 dirichlet_vals: None | np.ndarray = None,
                 neumann_inds: None | np.ndarray = None,
                 neumann_vals: None | np.ndarray = None,
                 robin_inds: None | np.ndarray = None,
                 robin_alpha: None | np.ndarray = None,
                 robin_beta: None | np.ndarray = None,
                 **kwargs: Any) -> np.ndarray:
    """
    Apply boundary conditions to a Laplacian field.

    Parameters
    ----------
    L : np.ndarray
        Laplacian values computed with periodic boundary conditions,
        same shape as the field.
    phi : np.ndarray
        scalar field, same shape as L. Only needed for Robin bc.
    free : None or np.ndarray
        boolean mask, same shape as L, True where no bc is applied.
    fixed : None or np.ndarray
        boolean mask, same shape as L, True at zero-Dirichlet nodes.
    dirichlet_inds : None or np.ndarray
        boolean mask, same shape as L, True at prescribed-value Dirichlet nodes.
    dirichlet_vals : None or np.ndarray
        prescribed values, same shape as L.
    neumann_inds : None or np.ndarray
        boolean mask, same shape as L, True at Neumann boundary nodes.
    neumann_vals : None or np.ndarray
        flux correction values (-2*g/dx), same shape as L.
    robin_inds : None or np.ndarray
        boolean mask, same shape as L, True at Robin boundary nodes.
    robin_alpha : None or np.ndarray
        field coefficient (-2*a/(b*dx)), same shape as L.
    robin_beta : None or np.ndarray
        constant term (2*g/(b*dx)), same shape as L.

    Returns
    -------
    L : np.ndarray
        Laplacian values with corrected boundary conditions.
    """
    # fixed: zero out so those nodes do not update during time-stepping
    if fixed is not None:
        L[fixed] = 0.
    # dirichlet: zero out at prescribed DOFs (caller enforces u = dirichlet_vals)
    if dirichlet_inds is not None:
        L[dirichlet_inds] = 0.
    # neumann: add flux correction at boundary DOFs
    if neumann_inds is not None and neumann_vals is not None:
        L[neumann_inds] += neumann_vals[neumann_inds]
    # robin: a*u + b*du/dn = g  →  correction = alpha*phi + beta
    # with alpha = -2*a/(b*dx), beta = 2*g/(b*dx)
    if robin_inds is not None and robin_alpha is not None and robin_beta is not None:
        L[robin_inds] += robin_alpha[robin_inds] * phi[robin_inds] + robin_beta[robin_inds]
    return L

def gradient_1d(f: np.ndarray,
                dx: float) -> np.ndarray:
    """
    Gradient operator in 1D on even spaced line with periodic boundary conditions.
    Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_x) or (ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing.

    Returns
    -------
    grad : np.ndarray shape (ngrid_x,1) or (ngrid_x,n_channel,1)
        gradient values.
    """
    return np.stack([np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)], axis=-1) / (2 * dx)


def gradient_2d(f: np.ndarray,
                dx: float) -> np.ndarray:
    """
    Gradient operator in 2D on rectangular even spaced grid with periodic
    boundary conditions. Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_y,ngrid_x) or (ngrid_y,ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing.

    Returns
    -------
    grad : np.ndarray shape (ngrid_y,ngrid_x,2) or (ngrid_y,ngrid_x,n_channel,2)
        gradient values.
    """
    return np.stack([np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0),
                     np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)], axis=-1) / (2 * dx)


def gradient_3d(f: np.ndarray,
                dx: float) -> np.ndarray:
    """
    Gradient operator in 3D on rectangular even spaced grid with periodic
    boundary conditions. Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x) or (ngrid_z,ngrid_y,ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing.

    Returns
    -------
    grad : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x,3) or (ngrid_z,ngrid_y,ngrid_x,n_channel,3)
        gradient values.
    """
    return np.stack([np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0),
                     np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1),
                     np.roll(f, -1, axis=2) - np.roll(f, 1, axis=2)], axis=-1) / (2 * dx)


def divergence_1d(v: np.ndarray,
                  dx: float) -> np.ndarray:
    """
    Divergence operator in 1D on even spaced line with periodic boundary conditions.
    Discretization is done via central difference.

    Parameters
    ----------
    v : np.ndarray shape (ngrid_x,1)
        vector field values.
    dx : float
        lattice spacing.

    Returns
    -------
    div : np.ndarray shape (ngrid_x)
        divergence values.
    """
    return (np.roll(v[..., 0], -1, axis=0) - np.roll(v[..., 0], 1, axis=0)) / (2 * dx)


def divergence_2d(v: np.ndarray,
                  dx: float) -> np.ndarray:
    """
    Divergence operator in 2D on rectangular even spaced grid with periodic
    boundary conditions. Discretization is done via central difference.

    Parameters
    ----------
    v : np.ndarray shape (ngrid_y,ngrid_x,2)
        vector field values.
    dx : float
        lattice spacing.

    Returns
    -------
    div : np.ndarray shape (ngrid_y,ngrid_x)
        divergence values.
    """
    return ((np.roll(v[..., 0], -1, axis=0) - np.roll(v[..., 0], 1, axis=0)) +
            (np.roll(v[..., 1], -1, axis=1) - np.roll(v[..., 1], 1, axis=1))) / (2 * dx)


def divergence_3d(v: np.ndarray,
                  dx: float) -> np.ndarray:
    """
    Divergence operator in 3D on rectangular even spaced grid with periodic
    boundary conditions. Discretization is done via central difference.

    Parameters
    ----------
    v : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x,3)
        vector field values.
    dx : float
        lattice spacing.

    Returns
    -------
    div : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x)
        divergence values.
    """
    return ((np.roll(v[..., 0], -1, axis=0) - np.roll(v[..., 0], 1, axis=0)) +
            (np.roll(v[..., 1], -1, axis=1) - np.roll(v[..., 1], 1, axis=1)) +
            (np.roll(v[..., 2], -1, axis=2) - np.roll(v[..., 2], 1, axis=2))) / (2 * dx)


def laplacian_1d(f: np.ndarray,
                 dx: float) -> np.ndarray:
    """
    Laplacian operator in 1D on even spaced line with periodic boundary conditions.
    Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_x) or (ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing

    Returns
    -------
    L : np.ndarray shape (ngrid_x) or (ngrid_x,n_channel)
        Laplacian values.
    """
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) - 2*f) / dx**2

def laplacian_2d(f: np.ndarray,
                 dx: float) -> np.ndarray:
    """
    Laplacian operator in 2D on rectangular even spaced grid with periodic boundary
    conditions. Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_y,ngrid_x) or (ngrid_y,ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing

    Returns
    -------
    L : np.ndarray shape (ngrid_y,ngrid_x) or (ngrid_y,ngrid_x,n_channel)
        Laplacian values.
    """
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +\
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4*f) / dx**2

def laplacian_3d(f: np.ndarray,
                 dx: float) -> np.ndarray:
    """
    Laplacian operator in 3D on rectangular even spaced grid with periodic boundary
    conditions. Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x) or (ngrid_z,ngrid_y,ngrid_x,n_channel)
        function values.
    dx : float
        lattice spacing

    Returns
    -------
    L : np.ndarray shape (ngrid_z,ngrid_y,ngrid_x) or (ngrid_z,ngrid_y,ngrid_x,n_channel)
        Laplacian values.
    """
    return (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +\
            np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +\
            np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6*f) / dx**2
