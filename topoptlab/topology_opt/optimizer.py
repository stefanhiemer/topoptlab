# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union
from functools import partial
from cProfile import Profile
from datetime import datetime
import inspect
#
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.ndimage import convolve
#
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# functions to create filters 
from topoptlab.filter.filter import TOFilter
from topoptlab.filter.workflows import prepare_filters,sensitivity_propagation,filter_design_variables
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import mbb_2d
# set up finite element problem
from topoptlab.fem import create_matrixinds,assemble_matrix,apply_bc
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d, lf_strain_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d, lf_strain_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d
# generic functions for solving phys. problem
from topoptlab.solve_linsystem import solve_lin
#
from topoptlab.material_interpolation import simp, simp_dx
# constrained optimizers
from topoptlab.optimizer.optimality_criterion import oc_top88,oc_mechanism,oc_generalized
from topoptlab.objectives import compliance
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel,\
                            check_simulation_params,dict_without,\
                            default_outputkw,check_output_kw,check_optimizer_kw,\
                            default_el_flags_policy,check_el_flags_policy,\
                            check_constraints,expand_eq_constraints
from topoptlab.objectives import vol_frac as volume_fraction_constraint
# logging related stuff
from topoptlab.log_utils import EmptyLogger,SimpleLogger
from topoptlab.convergence_criteria import max_design_change
from topoptlab.param_continuation import run_continuation
#
from mmapy import mmasub, gcmmasub, asymp, concheck, raaupdate
# 
from topoptlab.problem_solver import ProblemSolver
# 
from topoptlab.topology_opt.mesh import create_mesh
from topoptlab.topology_opt.history import initialize_history, update_history
from topoptlab.topology_opt.solver_routines import adjoint_loop, initialize_problems, solver_loop
from topoptlab.topology_opt.plotting import initialize_plotting

def initialize_optimizer(optimizer: str,
                         optimizer_kw: Dict,
                         x: np.ndarray,
                         xPhys: np.ndarray,
                         n_constr: int,
                         n_el: int,
                         ft,
                         el_flags,
                         prescribed_mask,
                         volfrac) -> Tuple[Dict, np.ndarray, np.ndarray, int]:
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