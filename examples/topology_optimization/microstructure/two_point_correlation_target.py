# SPDX-License-Identifier: GPL-3.0-or-later
"""
Microstructure design by matching a target two-point correlation function.

The two-point correlation at lag d is

    S2(d) = <x(r) * x(r+d)>_r  (periodic average)

It is a 2-D image (same size as the grid) with:
  - S2(0)    = phi          (volume fraction, excluded from objective)
  - S2(d->inf) -> phi^2    (ergodic random limit)

We target an isotropic exponential decay

    S2_target(d) = phi^2 + (phi - phi^2) * exp(-|d| / xi)

with correlation length xi (in grid units).  Any microstructure that
matches this profile is a valid solution; the density filter regularises
the spatial frequency content.

Usage
-----
    python two_point_correlation_target.py [nelx] [nely] [volfrac] [xi]
"""
import sys
import numpy as np

from topoptlab.topology_opt.driver import main
from topoptlab.objectives import two_point_correlation_image_squarederror, gray_indicator_constraint
from topoptlab.convergence_criteria import max_design_change

if __name__ == "__main__":
    nelx = 50
    nely = 50
    volfrac = 0.5
    rmin = 2.4
    xi = 5.  # correlation length in grid units

    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        xi = float(sys.argv[4])

    # Build target S2 image in FFT layout.
    # np.fft.fftfreq(n, 1/n) gives the signed integer lags
    # [0, 1, ..., n/2-1, -n/2, ..., -1], matching the output of irfftn.
    dy = np.fft.fftfreq(nely, 1. / nely)[:, None]   # (nely, 1)
    dx = np.fft.fftfreq(nelx, 1. / nelx)[None, :]   # (1, nelx)
    r  = np.sqrt(dy**2 + dx**2)                       # (nely, nelx)
    S2_target = volfrac**2 + (volfrac - volfrac**2) * np.exp(-r / xi)
    # S2_target shape: (nely, nelx)  — single-material, passed as-is
    rng = np.random.default_rng(seed=0)
    x0  = rng.random((nelx * nely, 1))
    #
    main(nelx=nelx,
         nely=nely,
         volfrac=volfrac,
         rmin=rmin,
         ft=1, # no filter
         filter_mode="matrix",
         problems=[], # no physical simulation
         optimizer="mma",
         initial_guess={"x": np.ones((nelx*nely,1))*volfrac - x0*0.01},
         obj_func=two_point_correlation_image_squarederror,
         obj_kw={"S2_target": S2_target, #"scale_factor": 2.
                 },
        constraints=[{"name": "gray indicator",
                       "func": gray_indicator_constraint,
                       "type": "leq",
                       "value": 0.5,
                       "kw": {}}],
         nouteriter=500,
         convergence_kw={"conv_tol": 1e-4,
                         "change_func": max_design_change},
         output_kw={"file": "two_point_corr_target",
                    "display": True,
                    "export": True,
                    "write_log": True,
                    "verbosity": 20,
                    "profile": False,
                    "output_movie": False,},)
