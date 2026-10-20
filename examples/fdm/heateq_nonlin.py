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


def _make_dirichlet_mask(ndim: int, grid_size: int) -> np.ndarray:
    """Bool mask that is True at every face node (all-faces Dirichlet)."""
    shape = ndim * [grid_size]
    mask = np.zeros(shape, dtype=bool)
    for ax in range(ndim):
        sl_lo = [slice(None)] * ndim; sl_lo[ax] = 0;            mask[tuple(sl_lo)] = True
        sl_hi = [slice(None)] * ndim; sl_hi[ax] = grid_size - 1; mask[tuple(sl_hi)] = True
    return mask


def generate_target_mass(ndim: int,
                         grid_size: int,
                         dx: float,
                         dt: float,
                         n_steps: int,
                         obs_steps: np.ndarray,
                         theta_true: np.ndarray,
                         c_surf: float = 1.0,
                         mass_factor: float = 1.0,
                         dtype=np.float64) -> np.ndarray:
    """
    Simulate solvent absorption into a dry material and return observed mass.

    A material, initially dry (c=0), is immersed in a solvent bath so that all
    surface nodes are held at c_surf (Dirichlet BC). The total uptake mass at
    each observation time is

        M(t) = mass_factor * dx**ndim * sum(c(x, t))

    Parameters
    ----------
    ndim : int
        Spatial dimensionality.
    grid_size : int
        Grid points per dimension.
    dx : float
        Lattice spacing.
    dt : float
        Time step size.
    n_steps : int
        Total number of forward time steps.
    obs_steps : array-like of int
        Time step indices at which mass is recorded.
    theta_true : np.ndarray shape (n_terms,)
        True polynomial diffusion coefficients D(c) = sum_k theta[k]*c^k.
    c_surf : float
        Fixed concentration at all boundary nodes (solvent bath value).
    mass_factor : float
        Converts sum(c)*dx^ndim to physical mass (e.g. molar mass / unit volume).
    dtype : dtype

    Returns
    -------
    mass_obs : np.ndarray shape (len(obs_steps),)
        Observed mass at each observation time step.
    """
    if ndim == 1:
        gradient, divergence = gradient_1d, divergence_1d
    elif ndim == 2:
        gradient, divergence = gradient_2d, divergence_2d
    else:
        gradient, divergence = gradient_3d, divergence_3d
    shape = ndim * [grid_size]
    vol = dx**ndim
    dirmask = _make_dirichlet_mask(ndim, grid_size)
    c = np.zeros(shape, dtype=dtype)
    c[dirmask] = c_surf
    obs_set = set(int(s) for s in obs_steps)
    mass_obs = []
    for step in range(n_steps):
        if step in obs_set:
            mass_obs.append(mass_factor * vol * c.sum())
        D_c = _eval_D(theta_true, c)
        flux = D_c[..., None] * gradient(c, dx)
        div_flux = divergence(flux, dx)
        c = c + dt * div_flux
        c[dirmask] = c_surf
    if n_steps in obs_set:
        mass_obs.append(mass_factor * vol * c.sum())
    return np.array(mass_obs, dtype=dtype)


def heateq_mass_absorption(ndim: int = 2,
                           grid_size: int = 32,
                           dx: float = 1.,
                           dt: float = 0.01,
                           theta0: List[float] = None,
                           Dmin: float = 0.,
                           Dmax: float = 10.,
                           n_steps: int = 200,
                           c_surf: float = 1.0,
                           mass_obs: np.ndarray = None,
                           obs_steps: np.ndarray = None,
                           mass_factor: float = 1.0,
                           n_iter: int = 200,
                           display: bool = True,
                           dtype: Callable = np.float64,
                           output_kw: Dict = None):
    """
    Calibrates polynomial diffusion coefficients from total-mass observations.

    Models a dry material soaked in a solvent bath: all surface nodes are held at
    c_surf (Dirichlet BC) and c=0 initially.  The only measurement is the total
    uptake mass at each observation time,

        M(t) = mass_factor * dx**ndim * sum(c(x, t))

    Minimizes  sum_i (M_sim(t_i) - M_obs(t_i))**2  via adjoint-based gradient
    descent over the polynomial diffusion coefficients theta.

    Parameters
    ----------
    ndim : int
    grid_size : int
    dx : float
    dt : float
    theta0 : list of float
        Initial coefficients [theta_0, theta_1, ...].
    Dmin, Dmax : float
        Box constraints applied per coefficient.
    n_steps : int
        Forward simulation time steps.
    c_surf : float
        Dirichlet value at all boundary nodes.
    mass_obs : np.ndarray shape (n_obs,)
        Measured total mass at each entry of obs_steps.
    obs_steps : array-like of int
        Time step indices of the mass measurements.
    mass_factor : float
        Physical constant: mass = mass_factor * dx**ndim * sum(c).
    n_iter : int
    display : bool
    dtype : callable
    output_kw : dict

    Returns
    -------
    c : np.ndarray
        Concentration field at the final time step.
    theta : np.ndarray shape (n_terms,)
        Optimised polynomial coefficients.
    theta_hist : list of np.ndarray
    """
    if theta0 is None:
        theta0 = [1.0]
    if output_kw is None:
        output_kw = default_outputkw()
    n_terms = len(theta0)
    obs_steps = np.asarray(obs_steps, dtype=int)
    mass_obs = np.asarray(mass_obs, dtype=dtype)
    obs_set = {int(s): i for i, s in enumerate(obs_steps)}
    log = SimpleLogger(file=output_kw["file"], verbosity=output_kw["verbosity"])
    log.info(f"number of spatial dimensions: {ndim}")
    log.info("elements: " + " x ".join([f"{grid_size}"] * ndim))
    log.info(f"n_obs: {len(obs_steps)}")
    if ndim == 1:
        gradient, divergence = gradient_1d, divergence_1d
    elif ndim == 2:
        gradient, divergence = gradient_2d, divergence_2d
    elif ndim == 3:
        gradient, divergence = gradient_3d, divergence_3d
    shape = ndim * [grid_size]
    vol = dx**ndim
    dirmask = _make_dirichlet_mask(ndim, grid_size)
    theta = np.array(theta0, dtype=dtype)
    dobj = np.zeros(n_terms, dtype=dtype)
    theta_old = None
    dobj_old = None
    theta_hist = []
    c_hist = np.zeros([n_steps + 1] + list(shape), order="F", dtype=dtype)
    c_hist[0][dirmask] = c_surf
    if display and ndim == 2:
        plt.ion()
        fig, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(10, 4))
        img_c = ax_c.imshow(c_hist[0], cmap='hot', origin='lower',
                            vmin=0., vmax=c_surf)
        ax_c.tick_params(axis='both', which='both',
                         bottom=False, left=False,
                         labelbottom=False, labelleft=False)
        ax_c.set_title("concentration")
        t_axis = obs_steps * dt
        line_obs, = ax_m.plot(t_axis, mass_obs, 'k--', label='observed')
        line_sim, = ax_m.plot(t_axis, np.zeros_like(mass_obs), 'r-', label='simulated')
        ax_m.set_xlabel("time")
        ax_m.set_ylabel("mass")
        ax_m.legend()
    for iteration in np.arange(n_iter):
        dobj[:] = 0.
        # --- forward ---
        for step in np.arange(n_steps):
            c = c_hist[step]
            D_c = _eval_D(theta, c)
            flux = D_c[..., None] * gradient(c, dx)
            div_flux = divergence(flux, dx)
            c_new = c + dt * div_flux
            c_new[dirmask] = c_surf
            c_hist[step + 1] = c_new
        # --- objective ---
        mass_sim = np.array([mass_factor * vol * c_hist[t].sum()
                             for t in obs_steps])
        obj = ((mass_sim - mass_obs)**2).sum()
        # --- adjoint ---
        # Because all boundary faces share the same Dirichlet value c_surf, the
        # PBC gradient/divergence operators are exact at interior nodes (opposite
        # faces wrap to the same c_surf value).  The adjoint BC is simply h=0 at
        # boundary nodes; no flux correction analogous to apply_finite_bc_div is
        # needed.
        h = np.zeros(shape, dtype=dtype)
        for nt in np.arange(n_steps - 1, -1, -1):
            # accumulate observation seed at t = nt+1
            if nt + 1 in obs_set:
                idx = obs_set[nt + 1]
                seed_val = 2. * (mass_sim[idx] - mass_obs[idx]) * mass_factor * vol
                h += seed_val  # uniform seed; boundary part zeroed on next line
            h[dirmask] = 0.   # Dirichlet adjoint BC
            # gradient w.r.t. theta[k]
            c_nt = c_hist[nt]
            grad_c = gradient(c_nt, dx)
            for k in range(n_terms):
                flux_k = (c_nt**k)[..., None] * grad_c
                dobj[k] += dt * (h * divergence(flux_k, dx)).sum()
            # propagate adjoint: (dA/dc)^T h
            D_c = _eval_D(theta, c_nt)
            Dprime_c = _eval_Dprime(theta, c_nt)
            grad_h = gradient(h, dx)
            adj_h = divergence(D_c[..., None] * grad_h, dx)
            adj_h -= Dprime_c * np.einsum('...i,...i->...', grad_h, grad_c)
            h = h + dt * adj_h
        # handle seed at t=0
        if 0 in obs_set:
            h += 2. * (mass_sim[obs_set[0]] - mass_obs[obs_set[0]]) * mass_factor * vol
        # --- update ---
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
        if display and ndim == 2:
            img_c.set_array(c_hist[-1])
            line_sim.set_ydata(mass_sim)
            fig.canvas.draw(); plt.pause(0.001)
        if np.linalg.norm(dobj) < 1e-8:
            break
    if display:
        plt.close()
    return c_hist[-1], theta, theta_hist


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
    mode = "pointwise"
    if len(sys.argv) > 1:
        mode = sys.argv[1]          # "pointwise" or "mass"
    if len(sys.argv) > 2:
        ndim = int(sys.argv[2])
    if len(sys.argv) > 3:
        n = int(sys.argv[3])
    if len(sys.argv) > 4:
        display = bool(int(sys.argv[4]))

    if mode == "pointwise":
        # --- pointwise concentration matching ---
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

    elif mode == "mass":
        # --- total-mass absorption (solvent soaking) ---
        # Observe mass at 20 evenly-spaced snapshots; mass_factor converts
        # mol/L * (dx*m)^ndim to grams (here left at 1 for demonstration).
        theta_true = np.array([1.5, 0.5])
        obs_steps = np.linspace(10, n_steps, 20, dtype=int)
        mass_obs = generate_target_mass(
            ndim=ndim, grid_size=n, dx=dx, dt=dt,
            n_steps=n_steps, obs_steps=obs_steps,
            theta_true=theta_true, c_surf=1.0, mass_factor=1.0)
        c_found, theta_found, theta_hist = heateq_mass_absorption(
            ndim=ndim, grid_size=n, dx=dx, dt=dt,
            theta0=[1.0, 0.0], Dmin=0.0, Dmax=5.0,
            n_steps=n_steps, c_surf=1.0,
            mass_obs=mass_obs, obs_steps=obs_steps, mass_factor=1.0,
            n_iter=300, display=display, output_kw=output_kw)
        print(f"theta_true:  {theta_true}")
        print(f"theta_found: {theta_found}")
        if ndim == 2:
            t_axis = obs_steps * dt
            fig, axs = plt.subplots(1, 3, figsize=(14, 4))
            axs[0].imshow(c_found, cmap='hot', origin='lower', vmin=0., vmax=1.)
            axs[0].set_title("final concentration")
            axs[0].tick_params(axis='both', which='both',
                               bottom=False, left=False,
                               labelbottom=False, labelleft=False)
            mass_found = generate_target_mass(
                ndim=ndim, grid_size=n, dx=dx, dt=dt,
                n_steps=n_steps, obs_steps=obs_steps,
                theta_true=theta_found, c_surf=1.0, mass_factor=1.0)
            axs[1].plot(t_axis, mass_obs, 'k--', label='observed')
            axs[1].plot(t_axis, mass_found, 'r-', label='found')
            axs[1].set_xlabel("time"); axs[1].set_ylabel("mass"); axs[1].legend()
            theta_hist_arr = np.array(theta_hist)
            for k in range(len(theta_true)):
                axs[2].plot(theta_hist_arr[:, k], color=f'C{k}', label=f"theta[{k}] found")
                axs[2].axhline(theta_true[k], color=f'C{k}', linestyle='--',
                               label=f"theta[{k}] true = {theta_true[k]:.2f}")
            axs[2].set_xlabel("iteration"); axs[2].set_ylabel("theta"); axs[2].legend()
            plt.tight_layout()
            plt.show()
            input("Press any key...")
