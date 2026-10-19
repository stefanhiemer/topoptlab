# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.fdm import (gradient_1d, gradient_2d, gradient_3d,
                            divergence_1d, divergence_2d, divergence_3d)
from topoptlab.optimizer.gradient_descent import gradient_descent
from topoptlab.log_utils import SimpleLogger
from topoptlab.utils import default_outputkw


def _eval_D(theta: np.ndarray, c: np.ndarray) -> np.ndarray:
    """D(c; theta) = sum_k theta[k] * c^k."""
    D = np.zeros_like(c)
    for k, tk in enumerate(theta):
        D = D + tk * c**k
    return D


def _eval_Dprime(theta: np.ndarray, c: np.ndarray) -> np.ndarray:
    """dD/dc = sum_{k>=1} k * theta[k] * c^(k-1)."""
    Dp = np.zeros_like(c)
    for k, tk in enumerate(theta):
        if k > 0:
            Dp = Dp + k * tk * c**(k - 1)
    return Dp


def generate_target(ndim: int,
                    grid_size: int,
                    dx: float,
                    dt: float,
                    t_obs: int,
                    theta_true: np.ndarray,
                    dtype=np.float64) -> np.ndarray:
    """
    Generate target concentration by forward FDM simulation with D_true(c).

    Uses a point-source initial condition at the origin corner so that
    concentration gradients are non-trivial in all directions.

    Parameters
    ----------
    ndim : int
        dimensionality of problem.
    grid_size : int
        grid points in each dimension.
    dx : float
        lattice spacing.
    dt : float
        time step size.
    t_obs : int
        observation time step.
    theta_true : np.ndarray shape (n_terms,)
        polynomial coefficients of the true diffusion: D(c) = sum_k theta[k]*c^k.
    dtype : dtype
        dtype for the concentration field.

    Returns
    -------
    c_t : np.ndarray shape (grid_size,)*ndim
        target concentration field at time t_obs * dt.
    """
    if ndim == 1:
        gradient, divergence = gradient_1d, divergence_1d
    elif ndim == 2:
        gradient, divergence = gradient_2d, divergence_2d
    else:
        gradient, divergence = gradient_3d, divergence_3d
    shape = ndim * [grid_size]
    base_coords = list(np.indices(shape))
    neumann_mask = np.zeros(shape, dtype=bool)
    bc_pairs = []
    for ax in range(ndim):
        for face, ghost_val in ((0, grid_size - 1), (grid_size - 1, 0)):
            face_mask = np.zeros(shape, dtype=bool)
            sl = [slice(None)] * ndim; sl[ax] = face; face_mask[tuple(sl)] = True
            neumann_mask |= face_mask
            ghost_coords = list(base_coords)
            ghost_c = base_coords[ax].copy(); ghost_c[face_mask] = ghost_val
            ghost_coords[ax] = ghost_c
            sign = -1 if face == 0 else 1
            bc_pairs.append((face_mask, tuple(ghost_coords), ax, sign))
    c = np.zeros(shape, dtype=dtype)
    c[tuple([0] * ndim)] = 1.
    for _ in range(t_obs):
        D_c = _eval_D(theta_true, c)
        flux = D_c[..., None] * gradient(c, dx)
        div_flux = divergence(flux, dx)
        div_flux = apply_finite_bc_div(div_flux, flux, dx, neumann_mask, bc_pairs)
        c = c + dt * div_flux
    return c


def apply_finite_bc_div(div_field: np.ndarray,
                        flux: np.ndarray,
                        dx: float,
                        neumann_mask: np.ndarray,
                        bc_pairs: list) -> np.ndarray:
    """
    Correct a PBC flux divergence for zero-flux Neumann BCs at boundary nodes.

    At each face (face_mask, ghost_coords, ax, sign), the correction

        nv[face_mask] += sign * (F_ax[face_mask] - F_ax[ghost][face_mask]) / (2*dx)

    replaces the PBC ghost slot with the symmetric-ghost value, enforcing zero
    normal flux derivative.  sign = -1 for faces at the low end of their axis
    (PBC ghost is in the backward stencil slot) and +1 for the high end.

    Parameters
    ----------
    div_field : np.ndarray
        divergence of flux computed with periodic boundary conditions.
    flux : np.ndarray shape (..., ndim)
        vector flux field with trailing direction axis.
    dx : float
        lattice spacing.
    neumann_mask : np.ndarray, bool, same shape as div_field
        True at every boundary node across all faces.
    bc_pairs : list of (face_mask, ghost_coords, ax, sign)
        face_mask    — bool array, same shape as div_field, True on this face.
        ghost_coords — tuple of ndim int arrays, each same shape as div_field.
        ax           — int, axis perpendicular to this face.
        sign         — int, -1 for face==0, +1 for face==grid_size-1.

    Returns
    -------
    div_field : np.ndarray
        divergence with zero-flux Neumann corrections applied.
    """
    nv = np.zeros_like(div_field)
    for face_mask, ghost_coords, ax, sign in bc_pairs:
        F_ax = flux[..., ax]
        nv[face_mask] += sign * (F_ax[face_mask] -
                                  F_ax[tuple(gc[face_mask] for gc in ghost_coords)]) / (2 * dx)
    div_field[neumann_mask] += nv[neumann_mask]
    return div_field


def heat_eq_fdm(ndim: int = 2,
                grid_size: int = 128,
                dx: float = 1.,
                dt: float = 0.01,
                theta0: List[float] = None,
                Dmin: float = 0.,
                Dmax: float = 10.,
                n_steps: int = 1000,
                c_t: np.ndarray = None,
                t_obs: int = None,
                n_iter: int = 200,
                display: bool = True,
                dtype: Callable = np.float64,
                output_kw: Dict = default_outputkw()):
    """
    Calibrates polynomial diffusion coefficients for the nonlinear heat equation

        dc/dt = div(D(c; theta) * grad(c)),   D(c; theta) = sum_k theta_k * c^k

    by minimizing (c(t_obs) - c_t)**2 / N via adjoint-based gradient descent.

    Parameters
    ----------
    ndim : int
        dimensionality of problem.
    grid_size : int
        grid points in each dimension.
    dx : float
        spatial step size.
    dt : float
        time step size.
    theta0 : list of float
        initial polynomial coefficients [theta_0, theta_1, ...].
    Dmin : float
        lower bound applied per coefficient.
    Dmax : float
        upper bound applied per coefficient.
    n_steps : int
        number of time steps.
    c_t : np.ndarray shape (grid_size,)*ndim
        target concentration field at observation time t_obs.
    t_obs : int
        time step index at which the observation is made.
    n_iter : int
        number of optimization iterations.
    display : bool
        activate plotting during optimization.
    dtype : callable
        dtype used for the concentration field.
    output_kw : dict
        logging output keywords.

    Returns
    -------
    c : np.ndarray shape (grid_size,)*ndim
        final concentration field.
    theta : np.ndarray shape (n_terms,)
        optimized polynomial coefficients.
    theta_hist : list of np.ndarray
        theta at each iteration.
    """
    if theta0 is None:
        theta0 = [1.0]
    n_terms = len(theta0)
    log = SimpleLogger(file=output_kw["file"], verbosity=output_kw["verbosity"])
    log.info(f"number of spatial dimensions: {ndim}")
    log.info("elements: " + " x ".join([f"{grid_size}"] * ndim))
    # fetch gradient and divergence operators
    if ndim == 1:
        gradient, divergence = gradient_1d, divergence_1d
    elif ndim == 2:
        gradient, divergence = gradient_2d, divergence_2d
    elif ndim == 3:
        gradient, divergence = gradient_3d, divergence_3d
    # precompute per-face (face_mask, ghost_coords, ax, sign) for zero-flux Neumann
    shape = ndim * [grid_size]
    base_coords = list(np.indices(shape))
    neumann_mask = np.zeros(shape, dtype=bool)
    bc_pairs = []
    for ax in range(ndim):
        for face, ghost_val in ((0, grid_size - 1), (grid_size - 1, 0)):
            face_mask = np.zeros(shape, dtype=bool)
            sl = [slice(None)] * ndim; sl[ax] = face; face_mask[tuple(sl)] = True
            neumann_mask |= face_mask
            ghost_coords = list(base_coords)
            ghost_c = base_coords[ax].copy(); ghost_c[face_mask] = ghost_val
            ghost_coords[ax] = ghost_c
            sign = -1 if face == 0 else 1
            bc_pairs.append((face_mask, tuple(ghost_coords), ax, sign))
    # design variables: polynomial coefficients theta
    theta = np.array(theta0, dtype=dtype)
    dobj = np.zeros(n_terms, dtype=dtype)
    theta_old = None
    dobj_old = None
    theta_hist = []
    # storage for forward trajectory
    c = np.zeros([n_steps + 1] + ndim * [grid_size], order="F", dtype=dtype)
    c[0][tuple([0] * ndim)] = 1.
    N = np.prod(shape, dtype=int)
    # plot
    if display and ndim == 2:
        plt.ion()
        fig, ax_c = plt.subplots()
        img_c = ax_c.imshow(c[0], cmap='hot', origin='lower')
        ax_c.tick_params(axis='both', which='both',
                         bottom=False, left=False,
                         labelbottom=False, labelleft=False)
        ax_c.set_title("c")
    # optimization loop
    for iteration in np.arange(n_iter):
        dobj[:] = 0.
        # forward time-stepping: c[step+1] = c[step] + dt * div_bc(D(c[step]) * grad(c[step]))
        for step in np.arange(n_steps):
            D_c = _eval_D(theta, c[step])
            flux = D_c[..., None] * gradient(c[step], dx)
            div_flux = divergence(flux, dx)
            div_flux = apply_finite_bc_div(div_flux, flux, dx, neumann_mask, bc_pairs)
            c[step + 1] = c[step] + dt * div_flux
        # plot
        if display and ndim == 2:
            img_c.set_array(c[-1])
            fig.canvas.draw()
            plt.pause(0.001)
        # objective and adjoint seed at t_obs
        r = c[t_obs] - c_t
        obj = (r**2).sum() / N
        h = (2. / N) * r
        # adjoint loop (backpropagate from t_obs to 0)
        for nt in np.arange(t_obs - 1, -1, -1):
            D_c = _eval_D(theta, c[nt])
            Dprime_c = _eval_Dprime(theta, c[nt])
            grad_c = gradient(c[nt], dx)    # (..., ndim)
            grad_h = gradient(h, dx)        # (..., ndim)
            # gradient w.r.t. theta[k]: dobj[k] += dt * <h, div_bc(c^k * grad(c))>
            for k in range(n_terms):
                flux_k = (c[nt]**k)[..., None] * grad_c
                dv_k = divergence(flux_k, dx)
                dv_k = apply_finite_bc_div(dv_k, flux_k, dx, neumann_mask, bc_pairs)
                dobj[k] += dt * (h * dv_k).sum()
            # exact adjoint: (I + dt * dA/dc)^T h
            # term 1: div_pbc(D(c) * grad(h))
            adj_h = divergence(D_c[..., None] * grad_h, dx)
            # term 2: -D'(c) * (grad_h . grad_c)  [from d/dc(D'(c)*delta_c*grad(c))]
            adj_h -= Dprime_c * np.einsum('...i,...i->...', grad_h, grad_c)
            # term 3: BC adjoint of div_bc(D(c) * grad(delta_c))
            # contributes -d/dx_ax(q) where q[face] = sign*h[face]*D(c[face])/(2dx),
            # q[ghost] -= sign*h[face]*D(c[ghost])/(2dx)
            for face_mask, ghost_coords, ax_bc, sign_bc in bc_pairs:
                ghost_flat = tuple(gc[face_mask] for gc in ghost_coords)
                q = np.zeros_like(h)
                q[face_mask] = sign_bc * h[face_mask] * D_c[face_mask] / (2 * dx)
                q[ghost_flat] -= sign_bc * h[face_mask] * D_c[ghost_flat] / (2 * dx)
                adj_h -= (np.roll(q, -1, axis=ax_bc) - np.roll(q, 1, axis=ax_bc)) / (2 * dx)
            # term 4: BC adjoint of div_bc(D'(c) * delta_c * grad(c))
            # contributes pointwise at face and ghost nodes
            for face_mask, ghost_coords, ax_bc, sign_bc in bc_pairs:
                ghost_flat = tuple(gc[face_mask] for gc in ghost_coords)
                grad_c_ax = grad_c[..., ax_bc]
                adj_h[face_mask] += (sign_bc * h[face_mask]
                                     * Dprime_c[face_mask]
                                     * grad_c_ax[face_mask] / (2 * dx))
                adj_h[ghost_flat] -= (sign_bc * h[face_mask]
                                      * Dprime_c[ghost_flat]
                                      * grad_c_ax[ghost_flat] / (2 * dx))
            h = h + dt * adj_h
        # gradient descent step
        theta_new = gradient_descent(
            x=theta, dobj=dobj, xmin=Dmin, xmax=Dmax,
            xold=theta_old, dobjold=dobj_old)
        theta_old, dobj_old = theta.copy(), dobj.copy()
        theta = theta_new
        theta_hist.append(theta.copy())
        log.info("it.: {0} obj.: {1:.10f} theta: {2} |grad|: {3:.2e}".format(
            iteration, obj,
            np.array2string(theta, precision=4, separator=', '),
            np.linalg.norm(dobj)))
        if np.linalg.norm(dobj) < 1e-8:
            break
    plt.close()
    return c[-1], theta, theta_hist


if __name__ == "__main__":
    import sys
    output_kw = default_outputkw()
    output_kw["file"] = "heateq_nonlin"
    ndim = 2
    n = 64
    dx = 0.5
    dt = 1e-2
    n_steps = 100
    t_obs = n_steps
    display = True
    if len(sys.argv) > 1:
        ndim = int(sys.argv[1])
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        display = bool(int(sys.argv[3]))
    # D(c) = 1.5 + 0.5*c  (linear nonlinear diffusion, n_terms=2)
    theta_true = np.array([1.5, 0.5])
    c_t = generate_target(ndim=ndim, grid_size=n, dx=dx, dt=dt,
                          t_obs=t_obs, theta_true=theta_true)
    c_found, theta_found, theta_hist = heat_eq_fdm(
        ndim=ndim, grid_size=n, dx=dx, dt=dt,
        theta0=[1.0, 0.0], Dmin=0.0, Dmax=5.0,
        n_steps=n_steps, c_t=c_t, t_obs=t_obs,
        n_iter=200, display=display, output_kw=output_kw)
    print(f"theta_true:  {theta_true}")
    print(f"theta_found: {theta_found}")
    if ndim == 2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(c_t, cmap='hot', origin='lower')
        axs[0].set_title(f"c_t  (theta_true={theta_true})")
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        axs[1].imshow(c_found, cmap='hot', origin='lower')
        axs[1].set_title("c found")
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        theta_hist_arr = np.array(theta_hist)
        for k in range(len(theta_true)):
            axs[2].plot(theta_hist_arr[:, k], color=f'C{k}', label=f"theta[{k}] found")
            axs[2].axhline(theta_true[k], color=f'C{k}', linestyle='--',
                           label=f"theta[{k}] true = {theta_true[k]:.2f}")
        axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("theta")
        axs[2].legend()
        plt.tight_layout()
        plt.show()
        input("Press any key...")
