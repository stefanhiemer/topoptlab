# SPDX-License-Identifier: GPL-3.0-or-later
"""
Visualise microstructure optimisation results: design, S2 difference, S2 target.

The two-point correlation is computed via the Wiener–Khinchin theorem
(rfft2), consistent with ``two_point_correlation_image_squarederror``.
"""
import numpy as np
import matplotlib.pyplot as plt

from topoptlab.objectives import gray_indicator_constraint


def plot(xPhys: np.ndarray,
         S2_target: np.ndarray,
         nelx: int,
         nely: int,
         xi: float) -> None:
    """
    Display the microstructure and the pointwise difference S2_found - S2_target.

    Parameters
    ----------
    xPhys : np.ndarray, shape (nel, 1)
        Physical densities after optimisation.
    S2_target : np.ndarray, shape (nely, nelx)
        Target two-point correlation in FFT layout.
    nelx, nely : int
        Grid dimensions.
    xi : float
        Correlation length used for the target (label only).
    """
    x_img = xPhys.reshape(nely, nelx)

    # S2 via Wiener–Khinchin, same as two_point_correlation_image_squarederror
    Xf = np.fft.rfft2(x_img)
    S2_found = np.fft.irfft2(np.conj(Xf) * Xf, s=(nely, nelx)).real / (nelx * nely)

    diff = np.fft.fftshift(S2_found - S2_target)

    gray_val = gray_indicator_constraint(xPhys)[0]
    vmax = np.abs(diff).max()
    S2_target_c = np.fft.fftshift(S2_target)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(x_img, cmap="gray_r", origin="lower", vmin=0, vmax=1)
    axes[0].set_title(
        f"Microstructure\nvf = {float(xPhys.mean()):.3f},  M = {gray_val:.3f}",
        fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"S2 difference  (found − target, ξ = {xi})", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(S2_target_c, cmap="hot", origin="lower",
                         vmin=0, vmax=S2_target_c.max())
    axes[2].set_title(f"S2 target  (ξ = {xi})", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("two_point_corr_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    input("Press any key...")
