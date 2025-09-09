# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt

def cahn_hilliard_fft(ndim=2, grid_size=128,
                      dx=1.0, dt=0.01,
                      gamma=1.0, M=1.0,
                      n_steps=1000,
                      display=True,
                      seed=0,
                      dtype=np.float32):
    """
    Solves the Cahn-Hilliard equation using a pseudo spectrail method via 
    operator split and explicit time integration.

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

    # wave number grid for FFT
    kx = np.fft.fftfreq(grid_size, d=dx) * 2 * np.pi
    if ndim == 2:
        ky = np.fft.rfftfreq(grid_size, d=dx) * 2 * np.pi
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = kx**2 + ky**2
    elif ndim == 3:
        ky = kx.copy()
        kz = np.fft.rfftfreq(grid_size, d=dx) * 2 * np.pi
        kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
    #
    mu = np.zeros(c.shape,dtype=dtype)
    for step in range(n_steps):
        # # compute chemical potential
        mu[:] = c**3 - c - gamma * np.fft.irfftn(-k2 * np.fft.rfftn(a=c))
        # update concentration
        c[:] += dt * M * np.fft.irfftn(-k2 * np.fft.rfftn(mu))
        # plot concentration field for 2D
        if ndim == 2 and step % (n_steps // 100) == 0 and display:
            plt.imshow(c, cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()
        if step % (n_steps // 100) == 0:
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

    return c
        
if __name__ == "__main__":
    #
    ndim = 2
    n = 128
    display=True
    #
    import sys
    if len(sys.argv)>1:
        ndim = int(sys.argv[1])
    if len(sys.argv)>2:
        n = int(sys.argv[2])
    if len(sys.argv)>3:
        display = bool(int(sys.argv[3]))
        
    cahn_hilliard_fft(
        ndim=ndim,
        grid_size=n,
        dx=1.0,
        dt=0.009,
        gamma=0.5,
        M=1.0,
        n_steps=int(1e4),
        seed=0,
        display=display
    )
