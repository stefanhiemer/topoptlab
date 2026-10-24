# SPDX-License-Identifier: GPL-3.0-or-later
"""
Microstructure design by matching a target two-point correlation function.

Same problem as ``two_point_correlation_target.py`` (identical nelx, nely,
volfrac, rmin, xi and S2_target), but phase 2 crisps up the design with a
smooth, volume-conserving Heaviside projection filter instead of tightening
a gray-level constraint.

Two-phase optimisation strategy
--------------------------------
Phase 1 — density filter (ft=1) with a loose gray constraint: establishes
  the spatial frequency content so the 2-PC structure emerges. Identical to
  phase 1 of ``two_point_correlation_target.py``.
Phase 2 — density filter + ``EtaProjectorXu2010`` Heaviside projection,
  warm-started from phase 1: the projection sharpness ``beta`` is ramped up
  via ``beta_scaling`` continuation (2 -> 64, doubling each trigger, i.e. at
  most 5 continuation steps) until the design is crisp. ``EtaProjectorXu2010``
  adapts its threshold eta every call to keep the volume fraction fixed, so
  no gray-level constraint is needed in this phase.

Usage
-----
    python two_point_correlation_target_heaviside.py [nelx] [nely] [volfrac] [xi]
"""
import sys
import numpy as np

from topoptlab.topology_opt.driver import main
from topoptlab.objectives import two_point_correlation_image_squarederror, gray_indicator_constraint
from topoptlab.convergence_criteria import max_design_change
from topoptlab.param_continuation import beta_scaling
from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.haeviside_projectors import EtaProjectorXu2010
from topoptlab.optimizer.mma_utils import mma_defaultkws

if __name__ == "__main__":
    nelx = 1000
    nely = 1000
    volfrac = 0.5
    rmin = 3.
    xi = 3.     # correlation length in grid units

    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        xi = float(sys.argv[4])

    # Build target S2 image in FFT layout.
    # np.fft.fftfreq(n, 1/n) gives signed integer lags matching irfftn output.
    dy = np.fft.fftfreq(nely, 1. / nely)[:, None]   # (nely, 1)
    dx = np.fft.fftfreq(nelx, 1. / nelx)[None, :]   # (1, nelx)
    r  = np.sqrt(dy**2 + dx**2)
    S2_target = volfrac**2 + (volfrac - volfrac**2) * np.exp(-r / xi)

    rng = np.random.default_rng(seed=0)
    x0  = rng.random((nelx * nely, 1))

    output_kw = {"file": "two_point_corr_target_heaviside",
                 "display": False,
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

    # phase 1 -- plain density filter for a fixed number of steps, just to
    # get the design away from the uniform volfrac start before handing it
    # to the Eta projector (at beta=1 on a near-uniform field, find_eta's
    # root search has nothing to lock onto and fails to converge).
    x1, xTilde1, xPhys1, obj1 = main(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        rmin=rmin,
        ft=1,                    # density filter
        filter_mode="matrix",
        problems=[],
        optimizer="mma",
        initial_guess={"x": np.ones((nelx * nely, 1)) * volfrac - x0 * 0.01},
        obj_func=two_point_correlation_image_squarederror,
        obj_kw={"S2_target": S2_target},
        constraints=[gray_constr],
        nouteriter=50,
        convergence_kw={"conv_tol": 1e-2,
                        "change_func": max_design_change},
        output_kw=output_kw)

    optimizer_kw = mma_defaultkws(n=nelx * nely, n_constr=2)
    optimizer_kw["move"] = 0.01

    x, xTilde, xPhys, obj = main(
        nelx=nelx,
        nely=nely,
        volfrac=volfrac,
        rmin=rmin,
        ft=[DensityFilter, EtaProjectorXu2010],
        filter_kw={"beta": 2., "volfrac": volfrac},
        filter_mode="matrix",
        problems=[],
        optimizer="gcmma",
        optimizer_kw=optimizer_kw,
        initial_guess={"x": xPhys1},
        obj_func=two_point_correlation_image_squarederror,
        obj_kw={"S2_target": S2_target},
        constraints=[gray_constr],
        continuation_kw={
            "funcs":    [beta_scaling],
            "func_kws": [{"beta_scale":  2.,
                          "beta_limit":  64.,
                          "beta_update": 35,
                          "state":       {},
                          "constr_idx":  1}],
            "stop_flag": [False]},
        nouteriter=3000,
        convergence_kw={"conv_tol": 5e-3,
                        "change_func": max_design_change},
        output_kw=output_kw)

    from plot_two_point_corr import plot
    plot(xPhys=xPhys, S2_target=S2_target, nelx=nelx, nely=nely, xi=xi)
