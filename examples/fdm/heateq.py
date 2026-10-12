# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
from typing import Callable, Dict

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.fdm import laplacian_2d, laplacian_3d
from topoptlab.optimizer.gradient_descent import gradient_descent
from topoptlab.log_utils import SimpleLogger
from topoptlab.utils import default_outputkw


def analytic_c_t(ndim: int,
                 grid_size: int,
                 dx: float,
                 dt: float,
                 t_obs: int,
                 D_true: float) -> np.ndarray:
    """
    Analytic target concentration for a unit peak at x=0 diffusing over time.

    Uses the method-of-images half-space Gaussian, valid when the diffusion
    length sqrt(4*D*T) is much smaller than the domain length (grid_size-1)*dx.
    The dx prefactor ensures sum(c_t) = 1, consistent with FDM mass conservation
    for an initial condition c[0]=1 at the boundary.

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
    D_true : float
        true diffusion coefficient used to generate the target.

    Returns
    -------
    c_t : np.ndarray shape (grid_size,)*ndim
        target concentration field at time t_obs * dt.
    """
    x = np.arange(grid_size) * dx
    T = t_obs * dt
    c_t_1d = (dx / np.sqrt(np.pi * D_true * T)) * np.exp(-x**2 / (4 * D_true * T))
    if ndim == 2:
        return np.tile(c_t_1d[:, None], (1, grid_size))
    elif ndim == 3:
        return np.tile(c_t_1d[:, None, None], (1, grid_size, grid_size))


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
    Calibrates a scalar diffusion coefficient for the heat equation by
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
        initial diffusion coefficient.
    Dmin : float
        lower bound for diffusion coefficient.
    Dmax : float
        upper bound for diffusion coefficient.
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
    D : float
        optimized diffusion coefficient.
    D_hist : list
        diffusion coefficient value at each iteration.
    """
    log = SimpleLogger(file=output_kw["file"],
                       verbosity=output_kw["verbosity"])
    log.info(f"number of spatial dimensions: {ndim}")
    log.info("elements: " + " x ".join([f"{grid_size}"] * ndim))
    cfl = D0 * dt / dx**2
    log.info(f"CFL = {cfl:.4f}  (stable if < {1 / (2 * ndim):.4f})")
    if cfl >= 1 / (2 * ndim):
        log.info("WARNING: CFL condition violated, explicit scheme is unstable.")
    # fetch laplacian operator
    pbc = ndim * [False]
    if ndim == 2:
        laplacian = partial(laplacian_2d, pbc=pbc)
    elif ndim == 3:
        laplacian = partial(laplacian_3d, pbc=pbc)
    # design variable: scalar diffusion coefficient
    D = np.array([[D0]], dtype=dtype)
    dobj = np.zeros((1, 1), dtype=dtype)
    Dold = None
    dobjold = None
    D_hist = []
    # storage for forward trajectory
    c = np.zeros([n_steps + 1] + ndim * [grid_size], order="F", dtype=dtype)
    # initial condition: unit peak at x=0 boundary
    c[0] = np.zeros(ndim * [grid_size], dtype=dtype)
    c[0][tuple([0] + [slice(None)] * (ndim - 1))] = 1.
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
        # forward time-stepping
        for step in np.arange(n_steps):
            c[step + 1] = c[step] + dt * D[0, 0] * laplacian(f=c[step], dx=dx)
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
            dobj[0, 0] += dt * (h * laplacian(f=c[nt], dx=dx)).sum()
            h = h + dt * D[0, 0] * laplacian(f=h, dx=dx)
        # gradient descent step
        Dnew = gradient_descent(x=D.flatten(),
                                dobj=dobj.flatten(),
                                xmin=Dmin,
                                xmax=Dmax,
                                xold=Dold.flatten() if Dold is not None else None,
                                dobjold=dobjold.flatten() if dobjold is not None else None)
        Dold, dobjold = D.copy(), dobj.copy()
        D = Dnew.reshape(1, 1)
        D_hist.append(D[0, 0])
        #
        log.info("it.: {0} obj.: {1:.10f} D.: {2:.6f} |grad|.: {3:.2e}".format(
                  iteration, obj, D[0, 0], abs(dobj[0, 0])))
        if abs(dobj[0, 0]) < 1e-8:
            break
    #
    plt.close()
    return c[-1], D[0, 0], D_hist


if __name__ == "__main__":
    import sys
    output_kw = default_outputkw()
    output_kw["file"] = "heateq"
    ndim = 2
    n = 256
    dx = 0.5
    dt = 5e-2
    n_steps = 100
    t_obs = n_steps
    D_true = 2.
    display = True
    if len(sys.argv) > 1:
        ndim = int(sys.argv[1])
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
    if len(sys.argv) > 3:
        display = bool(int(sys.argv[3]))
    c_t = analytic_c_t(ndim=ndim, grid_size=n, dx=dx, dt=dt,
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
    print(f"D_true:  {D_true:.6f}")
    print(f"D_found: {D_found:.6f}")
    if ndim == 2:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(c_t, cmap='hot', origin='lower')
        axs[0].set_title(f"c_t  (D_true={D_true})")
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        axs[1].imshow(c_found, cmap='hot', origin='lower')
        axs[1].set_title("c found")
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, left=False,
                           labelbottom=False, labelleft=False)
        axs[2].plot(D_hist, label="D found")
        axs[2].axhline(D_true, color='r', linestyle='--', label=f"D true = {D_true}")
        axs[2].set_xlabel("iteration")
        axs[2].set_ylabel("D")
        axs[2].legend()
        plt.tight_layout()
        plt.show()
        input("Press any key...")
