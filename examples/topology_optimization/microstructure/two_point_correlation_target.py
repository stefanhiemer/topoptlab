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
matches this profile is a valid solution.

Two-phase optimisation strategy
--------------------------------
Phase 1 — density filter (ft=1) with a loose gray constraint: establishes
  the spatial frequency content so the 2-PC structure emerges.
Phase 2 — no filter (ft=-1), warm-started from phase 1: the gray bound is
  progressively tightened via ``change_constraint`` continuation until the
  design is crisp (M < 0.01).

Usage
-----
    python two_point_correlation_target.py [nelx] [nely] [volfrac] [xi]
"""
import sys
import numpy as np

from topoptlab.topology_opt.driver import main
from topoptlab.objectives import two_point_correlation_image_squarederror, gray_indicator_constraint
from topoptlab.convergence_criteria import max_design_change
from topoptlab.param_continuation import change_constraint
from topoptlab.output_designs import threshold

if __name__ == "__main__":
    nelx   = 80
    nely   = 80
    volfrac = 0.5
    rmin   = 3.
    xi     = 3.     # correlation length in grid units

    if len(sys.argv) > 1:
        nelx   = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely   = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        xi     = float(sys.argv[4])

    # Build target S2 image in FFT layout.
    # np.fft.fftfreq(n, 1/n) gives signed integer lags matching irfftn output.
    dy = np.fft.fftfreq(nely, 1. / nely)[:, None]   # (nely, 1)
    dx = np.fft.fftfreq(nelx, 1. / nelx)[None, :]   # (1, nelx)
    r  = np.sqrt(dy**2 + dx**2)
    S2_target = volfrac**2 + (volfrac - volfrac**2) * np.exp(-r / xi)

    rng = np.random.default_rng(seed=0)
    x0  = rng.random((nelx * nely, 1))

    output_kw = {"file": "two_point_corr_target",
                 "display": True,
                 "export": False,
                 "write_log": True,
                 "verbosity": 20,
                 "profile": False,
                 "output_movie": False}

    # density filter
    gray_constr = {"name": "gray indicator",
                   "func": gray_indicator_constraint,
                   "type": "leq",
                   "value": 0.5,
                   "kw": {}}

    x, xTilde, xPhys, obj = main(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        rmin=rmin,
        ft=1,                   # density filter
        filter_mode="matrix",
        problems=[],
        optimizer="mma",
        initial_guess={"x": np.ones((nelx * nely, 1)) * volfrac - x0 * 0.01},
        obj_func=two_point_correlation_image_squarederror,
        obj_kw={"S2_target": S2_target},
        constraints=[gray_constr],
        nouteriter=400,
        convergence_kw={"conv_tol": 1e-2,
                        "change_func": max_design_change},
        output_kw=output_kw)
    #
    gray_constr["value"] = 0.1

    x, xTilde, xPhys, obj = main(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        rmin=rmin,
        ft=-1,                  # no filter — 2PC objective regularises scale
        filter_mode="matrix",
        problems=[],
        optimizer="mma",
        initial_guess={"x": threshold(xPhys, volfrac)},
        obj_func=two_point_correlation_image_squarederror,
        obj_kw={"S2_target": S2_target},
        constraints=[gray_constr],
        continuation_kw={
            "funcs":    [change_constraint],
            "func_kws": [{"constraint": gray_constr,
                          "target":     0.01,
                          "scale":      0.65,
                          "update":     35,
                          "constr_idx": 1}],
            "stop_flag": [False]},
        nouteriter=3000,
        convergence_kw={"conv_tol": 1e-2,
                        "change_func": max_design_change},
        output_kw=output_kw)

    from plot_two_point_corr import plot
    plot(xPhys=xPhys, S2_target=S2_target, nelx=nelx, nely=nely, xi=xi)
