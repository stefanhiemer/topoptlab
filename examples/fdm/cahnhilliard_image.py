# SPDX-License-Identifier: GPL-3.0-or-later
"""
Create a microstructure image from the standard Cahn-Hilliard model, e.g. as
input for two-point correlations and other statistical descriptors.

The final concentration c (approx. in [-1,1]) is mapped to a phase indicator
phi = (c+1)/2, clipped to [0,1] and written as

    - a greyscale PNG with one pixel per grid point (no axes, colorbar,
      title or interpolation; phi=1 -> white, phi=0 -> black). The PNG is
      written by matplotlib and has 4 identical channels (RGBA), so take one
      channel when reading it back, e.g. plt.imread(file)[..., 0],
    - a .npy file with the unquantized phi as float array.

Row 0 of the array is row 0 (top) of the PNG, so loading the PNG gives back
the same array layout. The grid is periodic, consistent with FFT based
correlations.

Usage:
    python cahnhilliard_image.py [n] [n_steps] [seed] [file]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from cahnhilliard_fdm import cahn_hilliard_fdm


def cahn_hilliard_image(grid_size: int = 128,
                        n_steps: int = 10000,
                        seed: int = 0,
                        file: str = "cahnhilliard_microstructure"
                        ) -> np.ndarray:
    """
    Run the 2D Cahn-Hilliard model and store the final field as greyscale
    image and as numpy array.

    Parameters
    ----------
    grid_size : int
        grid points in each dimension (= pixels per side).
    n_steps : int
        number of time steps. More steps give coarser structures.
    seed : int
        random seed for the initial condition.
    file : str
        file name without extension.

    Returns
    -------
    phi : np.ndarray of shape (grid_size,grid_size)
        phase indicator in [0,1].
    """
    c = cahn_hilliard_fdm(ndim=2, grid_size=grid_size,
                          dx=1., dt=0.02,
                          gamma=0.5, M=1.,
                          n_steps=n_steps,
                          seed=seed,
                          display=False,
                          dtype=np.float64)
    # map concentration to phase indicator
    phi = np.clip((c + 1.) / 2., 0., 1.)
    # one pixel per grid point, grey values fixed to the range [0,1]
    plt.imsave(file + ".png",
               phi,
               cmap="gray",
               vmin=0.,
               vmax=1.)
    return phi


if __name__ == "__main__":
    #
    n = 128
    n_steps = 10000
    seed = 0
    file = "cahnhilliard_microstructure"
    #
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_steps = int(sys.argv[2])
    if len(sys.argv) > 3:
        seed = int(sys.argv[3])
    if len(sys.argv) > 4:
        file = sys.argv[4]
    #
    cahn_hilliard_image(grid_size=n, n_steps=n_steps, seed=seed, file=file)
