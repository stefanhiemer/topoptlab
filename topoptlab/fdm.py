# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np


def bc_correction(f: np.ndarray,
                  dx: float,
                  ax: int,
                  neumann_mask_lo: np.ndarray,
                  flux_lo: np.ndarray,
                  neumann_mask_hi: np.ndarray,
                  flux_hi: np.ndarray) -> np.ndarray:
    """
    Neumann boundary correction for a finite-difference Laplacian along one axis.

    The ghost-point construction df/dn = g gives f_{-1} = f_1 - 2*dx*g, which
    adds a correction -2*g/dx at the low boundary and +2*g/dx at the high
    boundary. Only entries where the mask is True are corrected.

    Parameters
    ----------
    f : np.ndarray
        function values on the full grid.
    dx : float
        lattice spacing.
    ax : int
        axis along which the correction is applied.
    neumann_mask_lo : np.ndarray
        boolean mask on the low boundary face (shape of f with axis ax removed).
    flux_lo : np.ndarray
        prescribed flux df/dn at the low boundary face, same shape as mask.
    neumann_mask_hi : np.ndarray
        boolean mask on the high boundary face, same shape as mask.
    flux_hi : np.ndarray
        prescribed flux df/dn at the high boundary face, same shape as mask.

    Returns
    -------
    correction : np.ndarray shape same as f
        correction array to be added to the Laplacian.
    """
    correction = np.zeros_like(f)
    idx_lo = [slice(None)] * f.ndim
    idx_lo[ax] = 0
    idx_lo = tuple(idx_lo)
    idx_hi = [slice(None)] * f.ndim
    idx_hi[ax] = -1
    idx_hi = tuple(idx_hi)
    correction[idx_lo] -= (2.0 / dx) * np.where(neumann_mask_lo, flux_lo, 0.0)
    correction[idx_hi] += (2.0 / dx) * np.where(neumann_mask_hi, flux_hi, 0.0)
    return correction


def laplacian_2d(f: np.ndarray,
                 dx: float,
                 pbc: np.ndarray = None) -> np.ndarray:
    """
    Laplacian operator in 2D on rectangular even spaced grid.
    Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray 
        function values shape (ngrid_x,ngrid_y).
    dx : float
        lattice spacing.
    pbc : np.ndarray shape (2,), optional
        boolean array indicating periodic boundary conditions per axis.
        Defaults to [True, True]. For axes where pbc is False, zero-flux
        Neumann boundary conditions are applied.

    Returns
    -------
    laplacian : np.ndarray shape (ngrid_x,ngrid_y)
        laplacian values.
    """
    if pbc is None:
        pbc = [True, True]
    result = np.zeros_like(f)
    for ax in range(2):
        f_fwd = np.roll(f, 1, axis=ax)
        f_bwd = np.roll(f, -1, axis=ax)
        if not pbc[ax]:
            idx_lo = [slice(None), slice(None)]
            idx_lo[ax] = 0
            idx_hi = [slice(None), slice(None)]
            idx_hi[ax] = -1
            f_fwd[tuple(idx_lo)] = f[tuple(idx_lo)]
            f_bwd[tuple(idx_hi)] = f[tuple(idx_hi)]
        result += f_fwd + f_bwd - 2 * f
    return result / dx**2


def laplacian_3d(f: np.ndarray,
                 dx: float,
                 pbc: np.ndarray = None) -> np.ndarray:
    """
    Laplacian operator in 3D on rectangular even spaced grid.
    Discretization is done via central difference.

    Parameters
    ----------
    f : np.ndarray 
        function values shape (ngrid_x,ngrid_y,ngrid_z).
    dx : float
        lattice spacing.
    pbc : np.ndarray shape (3,), optional
        boolean array indicating periodic boundary conditions per axis.
        Defaults to [True, True, True]. For axes where pbc is False, zero-flux
        Neumann boundary conditions are applied.

    Returns
    -------
    laplacian : np.ndarray 
        laplacian values shape (ngrid_x,ngrid_y,ngrid_z).
    """
    if pbc is None:
        pbc = [True, True, True]
    result = np.zeros_like(f)
    for ax in range(3):
        f_fwd = np.roll(f, 1, axis=ax)
        f_bwd = np.roll(f, -1, axis=ax)
        if not pbc[ax]:
            idx_lo = [slice(None), slice(None), slice(None)]
            idx_lo[ax] = 0
            idx_hi = [slice(None), slice(None), slice(None)]
            idx_hi[ax] = -1
            f_fwd[tuple(idx_lo)] = f[tuple(idx_lo)]
            f_bwd[tuple(idx_hi)] = f[tuple(idx_hi)]
        result += f_fwd + f_bwd - 2 * f
    return result / dx**2
