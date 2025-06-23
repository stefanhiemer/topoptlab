from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

def cahn_hilliard_fd(ndim=2, grid_size=128,
                     dx=1.0, dt=0.01,
                     gamma=1.0, M=1.0,
                     n_steps=1000,
                     display=True):
    """
    Solves the Allen-Cahn equation using finite differences via operator split 
    and explicit time integration in periodic boundary conditions.

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
    n_steps : int 
        number of time steps

    Returns
    -------
    c : np.ndarray of shape (grid_size)*ndim
        final concentration.

    """
    warn("No sanity checks yet.")
    np.random.seed(0)
    # Grid and initial condition
    if ndim == 2:
        c = np.random.rand(grid_size, grid_size) * 0.01
    elif ndim == 3:
        c = np.random.rand(grid_size, grid_size, grid_size) * 0.01
    else:
        raise ValueError("Only 2D and 3D cases are supported.")
    c = c - c.mean()
    # Laplacian operator (finite difference)
    def laplacian(f):
        if ndim == 2:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f
            ) / dx**2
        elif ndim == 3:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
                np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6 * f
            ) / dx**2

    # Time-stepping loop
    mu = np.zeros(c.shape)
    for step in np.arange(n_steps):
        # update concentration
        c[:] += dt * M * ( gamma * laplacian(c) + c - c**3 )
        #
        print("time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                     dt*(step+1), c.min(), c.max(), np.mean(c) * dx**ndim))
        # Optional: Visualization for 2D
        if ndim == 2 and step % (n_steps // 100) == 0 and display:
            plt.imshow(c, cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()

    # Final visualization
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
    nndim = 2
    n = 128
    display=True
    #
    import sys
    if len(sys.argv)>1:
        nndim = int(sys.argv[1])
    if len(sys.argv)>2:
        n = int(sys.argv[2])
    if len(sys.argv)>3:
        display = bool(int(sys.argv[3]))
    # Run the simulation
    cahn_hilliard_fd(ndim=nndim, grid_size=n,
                     dx=1.0, dt=0.04,
                     gamma=0.5, M=2,
                     n_steps=int(1e5),
                     display=display)
