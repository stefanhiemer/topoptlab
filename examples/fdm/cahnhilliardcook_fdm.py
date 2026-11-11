# SPDX-License-Identifier: GPL-3.0-or-later
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.fdm import laplacian_2d,laplacian_3d

def cahn_hilliard_cook_fdm(ndim=2, grid_size=128,
                      dx=1.0, dt=0.01,
                      gamma=1.0, M=1.0, kT=1e-3,
                      n_steps=1000,
                      display=True,
                      seed=0,
                      dtype=np.float32):
    """
    Solves the Cahn-Hilliard-Cook equation (Cahn-Hilliard plus thermal noise,
    Cook 1970) using finite differences via operator split and explicit
    (Euler-Maruyama) time integration.

    The Cahn-Hilliard-Cook equation reads

        dc/dt = M lap(mu) + div(xi),   mu = c^3 - c - gamma lap(c)

    with Gaussian white noise xi satisfying the fluctuation-dissipation
    relation

        <xi_a(x,t) xi_b(x',t')> = 2 M kT delta_ab delta(x-x') delta(t-t').

    The noise enters as the divergence of a random vector field, so it
    conserves mass. It is discretized on a staggered grid: component a of xi
    lives at the midpoints i+1/2 between grid points along axis a, and the
    noise at grid point i is the discrete divergence

        div(xi)_i = sum_a (xi_a[i+1/2] - xi_a[i-1/2]) / dx.

    The contributions cancel pairwise when summed over the grid, so total mass
    is conserved exactly. The noise spectrum ~4 sin^2(k dx/2)/dx^2 matches the
    spectrum of the discrete 5/7-point laplacian, i.e. noise and dissipation
    are consistent on the grid. A collocated central difference
    (xi[i+1] - xi[i-1]) / (2 dx) would vanish at the highest resolvable wave
    number. Time integration is Euler-Maruyama, so the white noise variance
    per step and midpoint is 2 M kT / (dt dx^ndim). Periodic boundaries are
    assumed for the noise (np.roll).

    Parameters
    ----------
    ndim : int
        ndimensionality of problem.
    grid_size : int
        grid points in each ndimension.
    dx : float
        spatial step size.
    dt : float
        time step size.
    gamma : float 
        interfacial width
    M : float 
        mobility 
    kT : float
        dimensionless noise strength (thermal energy).
    n_steps : int 
        number of time steps

    Returns
    -------
    c : np.ndarray of shape (grid_size)*ndim
        final concentration.

    """
    # fetch laplacian operator
    if ndim == 2:
        laplacian = laplacian_2d
    elif ndim == 3:
        laplacian = laplacian_3d
    # set random seed
    np.random.seed(seed)
    # grid and initial condition
    if ndim == 2:
        c = np.random.rand(grid_size, grid_size) * 0.01
    elif ndim == 3:
        c = np.random.rand(grid_size, grid_size, grid_size) * 0.01
    else:
        raise ValueError("Only 2D and 3D cases are supported.")
    c = (c - c.mean()).astype(dtype)
    # time-stepping loop
    mu = np.zeros(c.shape,dtype=dtype)
    div_xi = np.zeros(c.shape,dtype=dtype)
    # standard deviation of noise per midpoint and time step
    sigma = np.sqrt(2 * M * kT / (dt * dx**ndim))
    for step in np.arange(n_steps):
        # compute chemical potential
        mu[:] =  c**3 - c - gamma * laplacian(f=c,dx=dx)
        # update concentration
        c[:] += dt * M * laplacian(f=mu,dx=dx)
        # Cook noise: divergence of staggered random vector field (see
        # docstring). xi[i] is located at the midpoint i+1/2, the rolled
        # array at i-1/2.
        div_xi[:] = 0.
        for ax in range(ndim):
            xi = sigma * np.random.standard_normal(c.shape).astype(dtype)
            div_xi += (xi - np.roll(xi, 1, axis=ax)) / dx
        c[:] += dt * div_xi
        # plot concentration field for 2D
        if ndim == 2 and (step+1) % (n_steps // 100) == 0 and display:
            plt.imshow(c, cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()
        if (step+1) % (n_steps // 100) == 0:
            print("time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                         dt*(step+1), c.min(), c.max(), np.mean(c) * dx**ndim))
    #
    print("final time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                 dt*(step+1), c.min(), c.max(), np.mean(c) * dx**ndim))
    # final plot concentration field
    if ndim == 2 and display:
        plt.imshow(c, cmap='RdBu', origin='lower')
        plt.colorbar(label='Concentration')
        plt.title("Final Step")
        plt.show()
    elif ndim == 3:
        print("Simulation complete. Use 3D visualization tools to analyze results.")

    return c
    
if __name__ == "__main__":
    #
    ndim = 2
    n = 64
    display=True
    #
    import sys
    if len(sys.argv)>1:
        ndim = int(sys.argv[1])
    if len(sys.argv)>2:
        n = int(sys.argv[2])
    if len(sys.argv)>3:
        display = bool(int(sys.argv[3]))
    # 
    cahn_hilliard_cook_fdm(ndim=ndim, grid_size=n,
                     dx=1., dt=0.02,
                     gamma=0.5, M=1., kT=1e-3,
                     n_steps=int(1e4),
                     display=display)
