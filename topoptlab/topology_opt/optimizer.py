# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
#
# functions to create filters 
# default application case that provides boundary conditions, etc.
# set up finite element problem
# different elements/physics
# generic functions for solving phys. problem
#
# constrained optimizers
# output final design to a Paraview readable format
# map element data to img/voxel
from topoptlab.utils import check_optimizer_kw
# logging related stuff
#
# 
# 

def initialize_optimizer(optimizer: str,
                         optimizer_kw: dict,
                         x: np.ndarray,
                         xPhys: np.ndarray,
                         n_constr: int,
                         n_el: int,
                         ft,
                         el_flags,
                         prescribed_mask,
                         volfrac) -> tuple[dict, np.ndarray, np.ndarray, int]:
    """
    Validate and set up optimizer state before the optimization loop.

    Calls ``check_optimizer_kw`` to fill defaults, sets ``max_history``, and
    for MMA/GCMMA clamps passive/active element bounds in ``optimizer_kw`` and
    adjusts ``x``/``xPhys`` accordingly.  For OC variants the Lagrange
    multiplier accumulator ``g`` is stored as ``optimizer_kw["g"]`` so callers
    do not need a separate variable.

    Returns
    -------
    optimizer_kw : dict
        Populated and (for MMA/GCMMA) bounds-adjusted optimizer parameters.
    x : np.ndarray
        Design variables, possibly clamped for passive/active elements.
    xPhys : np.ndarray
        Physical densities, possibly clamped for passive/active elements.
    max_history : int
        Number of previous iterates the optimizer requires.
    """
    optimizer_kw = check_optimizer_kw(optimizer=optimizer,
                                      n=x.shape[0],
                                      ft=ft,
                                      n_constr=n_constr,
                                      optimizer_kw=optimizer_kw)
    if optimizer in ["oc", "ocm"]:
        if n_constr != 1:
            raise ValueError(f"Optimizers 'oc' and 'ocm' support exactly one constraint, "
                             f"got {n_constr}.")
        optimizer_kw["g"] = 0
        max_history = 2
    elif optimizer == "ocg":
        optimizer_kw["g"] = 0
        max_history = 2
    elif optimizer in ["mma", "gcmma"]:
        max_history = 3
        if el_flags is not None:
            mask = el_flags == 1
            optimizer_kw["xmin"][mask] = 0.
            optimizer_kw["xmax"][mask] = 0. + 1e-9
            x[mask, 0] = 0.
            xPhys[mask, 0] = 0.
            mask = el_flags == 2
            optimizer_kw["xmin"][mask] = 1. - 1e-9
            optimizer_kw["xmax"][mask] = 1.
            x[mask, :] = 1.
            xPhys[mask, :] = 1.
            if volfrac is not None:
                x_free = np.clip(volfrac * n_el / (~prescribed_mask).sum(), 0., 1.)
                x[~prescribed_mask, 0] = x_free
                xPhys[~prescribed_mask, 0] = x_free
    else:
        raise ValueError("Unknown optimizer: ", optimizer)
    return optimizer_kw, x, xPhys, max_history