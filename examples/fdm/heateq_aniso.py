# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.fdm import (laplacian_bc,
                            gradient_1d, gradient_2d, gradient_3d,
                            divergence_1d, divergence_2d, divergence_3d)
from topoptlab.optimizer.gradient_descent import gradient_descent
from topoptlab.log_utils import SimpleLogger
from topoptlab.utils import default_outputkw


def generate_target(ndim: int,
                    grid_size: int,
                    dx: float,
                    dt: float,
                    t_obs: int,
                    D_true: np.ndarray,
                    dtype=np.float64) -> np.ndarray:
    """
    Generate target concentration by forward FDM simulation with D_true.

    Uses a point-source initial condition at the origin corner so that
    all components of the diffusion tensor are identifiable from the target.

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
    D_true : np.ndarray shape (ndim,ndim)
        true diffusion tensor.
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
    c[tuple([0] * ndim)] = 1.          # point source at origin corner
    for _ in range(t_obs):
        flux = np.einsum('ij,...j->...i', D_true, gradient(c, dx))
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

    For each face (face_mask, ghost_coords, ax, sign), the correction

        nv[face_mask] += sign * (F_ax[face_mask] - F_ax[ghost][face_mask]) / (2*dx)

    replaces the PBC ghost slot with the symmetric-ghost value (ghost = face),
    enforcing zero normal flux derivative.  sign = -1 for faces at the low end
    of their axis (where the PBC ghost occupies the backward stencil slot) and
    +1 for faces at the high end.

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
        sign         — int, -1 for face==0 (low end), +1 for face==grid_size-1.

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
                D0: float = 1.,
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
    Calibrates an anisotropic diffusion tensor for the heat equation by
    minimizing (c(t_obs) - c_t)**2 / N via adjoint-based gradient descent.

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
    D0 : float
        initial value on the diagonal of the diffusion tensor.
    Dmin : float
        lower bound for diffusion tensor entries.
    Dmax : float
        upper bound for diffusion tensor entries.
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
        dtype used for the temperature field.
    output_kw : dict
        logging output keywords.

    Returns
    -------
    c : np.ndarray shape (grid_size,)*ndim
        final temperature field.
    D : np.ndarray shape (ndim,ndim)
        optimized diffusion tensor.
    D_hist : list of np.ndarray
        diffusion tensor at each iteration.
    """
    log = SimpleLogger(file=output_kw["file"],
                       verbosity=output_kw["verbosity"])
    log.info(f"number of spatial dimensions: {ndim}")
    log.info("elements: " + " x ".join([f"{grid_size}"] * ndim))
    cfl = D0 * dt / dx**2
    log.info(f"CFL = {cfl:.4f}  (stable if < {1 / (2 * ndim):.4f})")
    if cfl >= 1 / (2 * ndim):
        log.info("WARNING: CFL condition violated, explicit scheme is unstable.")
    # fetch gradient and divergence operators
    if ndim == 1:
        gradient, divergence = gradient_1d, divergence_1d
    elif ndim == 2:
        gradient, divergence = gradient_2d, divergence_2d
    elif ndim == 3:
        gradient, divergence = gradient_3d, divergence_3d
    # precompute per-face (face_mask, ghost_coords, ax) triples for zero-flux Neumann
    shape = ndim * [grid_size]
    base_coords = list(np.indices(shape))   # ndim arrays, each shape (grid_size,)*ndim
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
    # design variable: anisotropic diffusion tensor (ndim x ndim)
    D = np.eye(ndim, dtype=dtype) * D0
    dobj = np.zeros((ndim, ndim), dtype=dtype)
    Dold = None
    dobjold = None
    D_hist = []
    # storage for forward trajectory
    c = np.zeros([n_steps + 1] + ndim * [grid_size], order="F", dtype=dtype)
    # initial condition: point source at origin corner
    c[0] = np.zeros(ndim * [grid_size], dtype=dtype)
    c[0][tuple([0] * ndim)] = 1.
    #
    N = np.prod(c[t_obs].shape, dtype=int)
    # plot stuff
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
        # forward time-stepping: c[step+1] = c[step] + dt * div(D @ grad(c[step]))
        for step in np.arange(n_steps):
            flux = np.einsum('ij,...j->...i', D, gradient(c[step], dx))
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
        # dobj[p,q] = dt * <h, div_bc(e_p * grad(c)[q])>  — BC-consistent gradient
        # hess_c[...,q,p] = ∂(grad_c_q)/∂x_p, with BC correction at the two faces perp to p
        for nt in np.arange(t_obs - 1, -1, -1):
            grad_c = gradient(c[nt], dx)                      # (..., ndim)
            hess_c = gradient(grad_c, dx)                     # (..., ndim_q, ndim_p): PBC
            for ax in range(ndim):
                for face, ghost_val in ((0, grid_size - 1), (grid_size - 1, 0)):
                    sign = -1 if face == 0 else 1
                    sf = tuple(face      if i == ax else slice(None) for i in range(ndim))
                    sg = tuple(ghost_val if i == ax else slice(None) for i in range(ndim))
                    corr = sign * (grad_c[sf] - grad_c[sg]) / (2 * dx)  # (...face, ndim_q)
                    hess_c[sf + (slice(None), ax)] += corr               # view, in-place
            dobj += dt * np.einsum('n,nqp->pq',
                                   h.reshape(-1),
                                   hess_c.reshape(-1, ndim, ndim))
            # exact adjoint: div_pbc(D^T grad h) + adj of bc correction
            grad_h = gradient(h, dx)
            flux_h = np.einsum('ji,...j->...i', D, grad_h)   # D.T @ grad(h)
            adj_h = divergence(flux_h, dx)
            # adjoint of apply_finite_bc_div: for each bc_pair build sparse r,
            # then subtract (D @ grad_pbc(r))[..., ax]
            for face_mask, ghost_coords, ax_bc, sign_bc in bc_pairs:
                r = np.zeros_like(h)
                r[face_mask] = sign_bc * h[face_mask] / (2 * dx)
                r[tuple(gc[face_mask] for gc in ghost_coords)] -= sign_bc * h[face_mask] / (2 * dx)
                grad_r = gradient(r, dx)
                adj_h -= np.einsum('ij,...j->...i', D, grad_r)[..., ax_bc]
            h = h + dt * adj_h
        # gradient descent step
        Dnew = gradient_descent(x=D.flatten(),
                                dobj=dobj.flatten(),
                                xmin=Dmin,
                                xmax=Dmax,
                                xold=Dold.flatten() if Dold is not None else None,
                                dobjold=dobjold.flatten() if dobjold is not None else None)
        Dold, dobjold = D.copy(), dobj.copy()
        D = Dnew.reshape(ndim, ndim)
        D_hist.append(D.copy())
        #
        log.info("it.: {0} obj.: {1:.10f} |D|.: {2:.6f} |grad|.: {3:.2e}".format(
                  iteration, obj, np.linalg.norm(D), np.linalg.norm(dobj)))
        if np.linalg.norm(dobj) < 1e-8:
            break
    #
    plt.close()
    return c[-1], D, D_hist

if __name__ == "__main__":
    import sys
    output_kw = default_outputkw()
    output_kw["file"] = "heateq_aniso"
    ndim = 2
    n = 256
    dx = 0.5
    dt = 5e-2
    n_steps = 100
    t_obs = n_steps
    display = True
    if len(sys.argv) > 1:
        ndim = int(sys.argv[1])
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        display = bool(int(sys.argv[3]))
    if ndim == 1:
        D_true = np.array([[2.]])
    elif ndim == 2:
        D_true = np.array([[2., 0.5], [0.5, 1.5]])
    else:
        D_true = np.array([[2., 0.3, 0.1], [0.3, 1.5, 0.2], [0.1, 0.2, 1.]])
    c_t = generate_target(ndim=ndim, grid_size=n, dx=dx, dt=dt,
                          t_obs=t_obs, D_true=D_true)
    c_found, D_found, D_hist = heat_eq_fdm(ndim=ndim, grid_size=n,
                                            dx=dx, dt=dt,
                                            D0=1.0, Dmin=0.1, Dmax=5.0,
                                            n_steps=n_steps,
                                            c_t=c_t,
                                            t_obs=t_obs,
                                            n_iter=200,
                                            display=display,
                                            output_kw=output_kw)
    print(f"D_true:\n{D_true}")
    print(f"D_found:\n{D_found}")
    if ndim == 2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(c_t, cmap='hot', origin='lower')
        axs[0].set_title(f"c_t  (D_true diag={np.diag(D_true)})")
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        axs[1].imshow(c_found, cmap='hot', origin='lower')
        axs[1].set_title("c found")
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        D_norms = [np.linalg.norm(d) for d in D_hist]
        axs[2].plot(D_norms, label="|D| found")
        axs[2].axhline(np.linalg.norm(D_true), color='r', linestyle='--',
                       label=f"|D| true = {np.linalg.norm(D_true):.3f}")
        axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("|D|")
        axs[2].legend()
        plt.tight_layout()
        plt.show()
        input("Press any key...")
