# SPDX-License-Identifier: GPL-3.0-or-later
"""
Two- and three-point correlations of a Cahn-Hilliard microstructure image
(created by examples/fdm/cahnhilliard_image.py).

The grey value of the image is used as phase indicator x_1 in [0,1] and the
complement x_0 = 1 - x_1 as second phase, so every correlation is a tensor
over the phase combinations. The correlations are calculated with the same
functions used for the microstructure optimization in this folder:

    - two_point_correlation with rcut=None, i.e. via FFT over the whole
      sample,
    - n_point_correlation with n=3 and a short cut-off rcut, i.e. collinear
      triplets (r, r+d, r+2d) with 0 < |d| <= rcut.

Both assume periodic boundaries and statistical isotropy (iso=True). In
addition the full two-point correlation map S2_11(d) of phase 1 and its
radial average are plotted.

Usage:
    python correlations_cahnhilliard.py [file] [rcut]
"""
import sys
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from topoptlab.objectives import two_point_correlation, n_point_correlation
from topoptlab.utils import map_eltoimg, map_imgtoel


def correlations(file: str = "cahnhilliard_microstructure.png",
                 rcut: float = 2.) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the greyscale image and calculate two- and three-point correlations.

    Parameters
    ----------
    file : str
        greyscale PNG written by matplotlib (RGBA with identical channels).
    rcut : float
        cut-off radius for the three-point correlation in pixels.

    Returns
    -------
    S2 : np.ndarray, shape (2,2)
        two-point correlation tensor.
    S3 : np.ndarray, shape (2,2,2)
        three-point correlation tensor.
    """
    # first channel, row 0 of the array is the top of the image
    img = plt.imread(file)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float64)
    nely, nelx = img.shape
    # phases as channels: 0 -> black/complement, 1 -> white
    xPhys = map_imgtoel(np.stack([1. - img, img], axis=-1),
                        nelx=nelx, nely=nely)
    # mappings as expected by the correlation functions
    mapping = lambda x: map_eltoimg(x, nelx=nelx, nely=nely)
    invmapping = lambda img: map_imgtoel(img, nelx=nelx, nely=nely)
    #
    S2 = two_point_correlation(xPhys=xPhys, rcut=None,
                               mapping=mapping, invmapping=invmapping)[0]
    S3 = n_point_correlation(xPhys=xPhys, rcut=rcut, n=3,
                             mapping=mapping, invmapping=invmapping)[0]
    #
    print(f"image: {file}, {nely}x{nelx}, phase fractions: "
          f"{xPhys.mean(axis=0).round(4)}")
    print("two-point correlation (FFT, whole sample):")
    for c in product(range(2), repeat=2):
        print(f"  S2{c} = {S2[c]:.6f}")
    print(f"three-point correlation (collinear, rcut = {rcut}):")
    for c in product(range(2), repeat=3):
        print(f"  S3{c} = {S3[c]:.6f}")
    # full S2 map of phase 1 via Wiener-Khinchin (lag 0 at the centre)
    X = np.fft.rfft2(img)
    S2_map = np.fft.irfft2(np.conj(X) * X, s=img.shape) / img.size
    S2_map = np.fft.fftshift(S2_map)
    # radial average
    dy, dx = np.indices(img.shape)
    r = np.hypot(dy - nely // 2, dx - nelx // 2).round().astype(int)
    S2_r = np.bincount(r.ravel(), S2_map.ravel()) / np.bincount(r.ravel())
    rmax = min(nely, nelx) // 2
    #
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("microstructure")
    axes[0].axis("off")
    im = axes[1].imshow(S2_map, cmap="viridis")
    axes[1].set_title(r"$S_2^{11}(\mathbf{d})$, lag 0 at centre")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    phi = img.mean()
    axes[2].plot(np.arange(rmax + 1), S2_r[:rmax + 1], "k-")
    axes[2].axhline(phi, ls="--", c="gray", label=r"$\phi$")
    axes[2].axhline(phi**2, ls=":", c="gray", label=r"$\phi^2$")
    axes[2].set_xlabel("|d| (pixels)")
    axes[2].set_ylabel(r"$S_2^{11}$")
    axes[2].set_title("radial average")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig("correlations_cahnhilliard.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    return S2, S3


if __name__ == "__main__":
    #
    file = "cahnhilliard_microstructure.png"
    rcut = 2.
    #
    if len(sys.argv) > 1:
        file = sys.argv[1]
    if len(sys.argv) > 2:
        rcut = float(sys.argv[2])
    #
    correlations(file=file, rcut=rcut)
