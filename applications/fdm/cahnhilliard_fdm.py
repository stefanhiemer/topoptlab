from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

def cahn_hilliard_fd(dim=2, grid_size=128,
                     dx=1.0, dt=0.01,
                     gamma=1.0, M=1.0,
                     n_steps=1000,
                     display=True,
                     seed=0,
                     dtype=np.float32):
    """
    Solves the Cahn-Hilliard equation using finite differences via operator 
    split and explicit time integration.

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
    # Laplacian operator (finite difference)
    def laplacian(f):
        if dim == 2:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) - 4 * f
            ) / dx**2
        elif dim == 3:
            return (
                np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) +
                np.roll(f, 1, axis=1) + np.roll(f, -1, axis=1) +
                np.roll(f, 1, axis=2) + np.roll(f, -1, axis=2) - 6 * f
            ) / dx**2
    # set random seed
    np.random.seed(seed)
    # Grid and initial condition
    if dim == 2:
        c = np.random.rand(grid_size, grid_size) * 0.01
    elif dim == 3:
        c = np.random.rand(grid_size, grid_size, grid_size) * 0.01
    else:
        raise ValueError("Only 2D and 3D cases are supported.")
    c = (c - c.mean()).astype(dtype)
    # Time-stepping loop
    mu = np.zeros(c.shape,dtype=dtype)
    for step in np.arange(n_steps):
        # compute chemical potential
        mu[:] =  c**3 - c - gamma * laplacian(c)
        # update concentration
        c[:] += dt * M * laplacian(mu)
        #
        # Optional: Visualization for 2D
        if dim == 2 and step % (n_steps // 100) == 0 and display:
            print("time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                         dt*(step+1), c.min(), c.max(), np.mean(c) * dx**dim))
            plt.imshow(c, cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()
    print(mu.dtype,c.dtype)
    print("final time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                 dt*(step+1), c.min(), c.max(), np.mean(c) * dx**dim))
    # Final visualization
    if dim == 2 and display:
        plt.imshow(c, cmap='RdBu', origin='lower')
        plt.colorbar(label='Concentration')
        plt.title("Final Step")
        plt.show()
    elif dim == 3:
        print("Simulation complete. Use 3D visualization tools to analyze results.")

    return c

def run_simulation(seed, gamma=0.5):
    c = cahn_hilliard_fd(
                         dim=ndim,
                         grid_size=n,
                         dx=1.0,
                         dt=0.04 * 0.5/gamma,
                         gamma=gamma,
                         M=1.0,
                         n_steps=int(1e2),
                         display=False,
                         seed=seed,
                         dtype=np.float32)
    #
    np.savetxt(f"runs/cahn-hilliard-gamma-{gamma}_{seed}.csv", c,
               header=f'# gamma {gamma}')
    return 

if __name__ == "__main__":
    #
    ndim = 2
    n = 256
    display=False
    #
    import sys
    if len(sys.argv)>1:
        ndim = int(sys.argv[1])
    if len(sys.argv)>2:
        n = int(sys.argv[2])
    if len(sys.argv)>3:
        display = bool(int(sys.argv[3]))
    # 
    #cahn_hilliard_fd(dim=ndim, grid_size=n,
    #                 dx=1.0, dt=0.04,
    #                 gamma=0.5, M=1.0,
    #                 n_steps=int(1e5),
    #                 display=display)
    
    #
    from functools import partial
    seeds = range(960)
    for gamma in [0.5,1.,2.,3.,4.]:
        
        simul = partial(run_simulation,gamma=gamma)
        with Pool(24) as pool:
            pool.map(simul, seeds)
