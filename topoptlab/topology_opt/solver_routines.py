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
from topoptlab.log_utils import BaseLogger
# 
from topoptlab.problem_solver import ProblemSolver
from topoptlab.linear_solvers import res_norm
from topoptlab.block_assembly import (assemble_global_block_system,
                                      solve_global_block_system,
                                      split_global_solution_by_fields,
                                      transpose_blocks)

def copy_relevant_fields(state: Dict, problem) -> Dict:
    """Shallow-copy only the fields written by solvers in problem."""
    solvers = [problem] if isinstance(problem, ProblemSolver) else problem
    keys = {k for solver in solvers for k in solver.output_keys}
    return {k: state[k].copy() for k in keys if k in state}

def converged(old: Dict, new: Dict, atol: float = 1e-8) -> bool:
    """True when the 2-norm of the field change across all keys in old is below atol."""
    r = np.concatenate([new[k].ravel() - old[k].ravel() for k in old])
    return res_norm(r, atol=atol)

def initialize_problems(problems: List, 
                        bcs: List[Callable], 
                        mesh_kw: Dict, 
                        solver_kw: List[Dict],
                        logger: Union[None,BaseLogger] = None
                        ) -> None:

    if callable(bcs):
        bcs = [bcs]
    if not (isinstance(bcs,list) and (len(bcs) == len(problems))):
        raise ValueError("Each problem needs a boundary condition. ",
                         "len(bcs) != len(problems): ", 
                         len(bcs), len(problems))
    for i,problem in enumerate(problems):
        #
        if isinstance(problem, list) and all([callable(item) for item in problem]):
            problems[i] = [solver(**mesh_kw,
                                  **solver_kw[i], 
                                  bc=bcs[i],
                                  logger=logger) \
                            for solver in problem]
        elif callable(problem):
            problems[i] = problem(**mesh_kw,
                                  **solver_kw[i], 
                                  bc=bcs[i], 
                                  logger=logger)
        else:
            raise TypeError("Do not initialize the solvers before handing them to main().")
        #
        if isinstance(problems[i], ProblemSolver):
            pass 
        elif isinstance(problems[i], list) and \
                all([isinstance(solver, ProblemSolver) \
                    for solver in problems[i]]):
            pass
        else:
            raise TypeError("After initialization problems should be a list of ProblemSolvers or ",
                            f"a list-of-lists of ProblemSolvers. type at index {i}: ",
                            type(problems[i]))

    return problems

def solver_loop(problems: List,
                state: Dict,
                ntimesteps: int,
                lin_solver_kw: List[Dict],
                preconditioner_kw: List[Dict],
                nproblem_solves: int = 100,
                coupling: List = None,
                parameters: Dict = {},
                solver_kw: List[Dict] = None,
                logger: Union[None, BaseLogger] = None,
                ) -> Dict:
    """
    Time-stepping loop over weakly or strongly coupled problem groups.

    problems : List
        Sequence of problem groups. Each entry is either a single
        ``ProblemSolver`` or a list of solvers forming a coupled group.
    state : Dict
        Initial field state; updated in place each time step.
    ntimesteps : int
        Number of time steps.
    nproblem_solves : int
        Maximum Picard iterations for ``"strong"`` coupling.
    coupling : List
        One entry per group in ``problems``:
        ``"weak"``        – solve each solver once in sequence.
        ``"strong"``      – partitioned Picard iteration until convergence.
        ``"monolithic"``  – assemble and solve one global block system
                            (implicit solvers only).
    parameters : Dict
        Design variables / fixed problem parameters (read-only).
    context : Dict
        Solver metadata forwarded to every solver; augmented with
        ``"tstep"`` and ``"solve_iter"`` at runtime.
    """
    # TO DO: check that all lists have same length
    #
    for tstep in np.arange(ntimesteps):
        # iterate problem groups in forward order
        for i, problem in enumerate(problems):
            #
            if coupling[i] == "weak":
                # single solver or list of solvers solved once in sequence
                solvers = [problem] if isinstance(problem, ProblemSolver) else problem
                for solver in solvers:
                    if solver.implicit:
                        system = solver.assemble(state,
                                                 parameters,
                                                 solver_kw[i],
                                                 lin_solver_kw[i],
                                                 logger)
                        out = solver.solve(system,
                                           state,
                                           parameters,
                                           solver_kw[i],
                                           lin_solver_kw[i],
                                           preconditioner_kw[i],
                                           logger)
                    else:
                        out = solver.solve({},
                                           state,
                                           parameters,
                                           solver_kw[i],
                                           lin_solver_kw[i],
                                           preconditioner_kw[i],
                                           logger)
                    state.update(out)
            #
            elif coupling[i] == "strong":
                # partitioned Picard iteration
                for solve_iter in np.arange(nproblem_solves):
                    state_old = copy_relevant_fields(state, problem)
                    for solver in problem:
                        if solver.implicit:
                            system = solver.assemble(state,
                                                     parameters,
                                                     solver_kw[i],
                                                     lin_solver_kw[i],
                                                     logger)
                            out    = solver.solve(system,
                                                  state,
                                                  parameters,
                                                  solver_kw[i],
                                                  lin_solver_kw[i],
                                                  preconditioner_kw[i],
                                                  logger)
                        else:
                            out = solver.solve({},
                                               state,
                                               parameters,
                                               solver_kw[i],
                                               lin_solver_kw[i],
                                               preconditioner_kw[i],
                                               logger)
                        state.update(out)
                    if converged(state_old, state):
                        break
            #
            elif coupling[i] == "monolithic":
                system = {}
                for solver in problem:
                    system.update(solver.assemble(state,
                                                  parameters,
                                                  solver_kw[i],
                                                  lin_solver_kw[i],
                                                  logger))
                global_system = assemble_global_block_system(system=system,
                                                             state=state,
                                                             parameters=parameters,
                                                             solver_kw=solver_kw[i],
                                                             logger=logger)
                solution = solve_global_block_system(global_system)
                state.update(split_global_solution_by_fields(solution, problem))
            #
            else:
                raise ValueError(f"Unknown coupling mode: {coupling[i]}")
    return state

def adjoint_loop(problems: List,
                 state: Dict,
                 adj_rhs: Dict,
                 ntimesteps: int,
                 nproblem_solves: int = 100,
                 coupling: Union[None,str,List[str]] = None,
                 parameters: Dict = {},
                 solver_kw: List[Dict] = None,
                 lin_solver_kw: List[Dict] = None,
                 preconditioner_kw: List[Dict] = None,
                 logger = None,
                 ) -> Dict:
    """
    Adjoint pass mirroring ``solver_loop``, walked backward in time and
    problem-group order.  Each solver's cached forward state is reused so
    no reassembly is needed for linear problems.

    problems : list
        Same sequence of problem groups as in the forward pass.
    state : dict
        Forward solution state (read-only; provides cached factorizations).
    adj_rhs : dict
        Seed sensitivities dL/d(field) for each output field of the last
        forward step; updated with cross-physics coupling terms in-place.
    ntimesteps : int
        Number of time steps (adjoint runs from ``ntimesteps-1`` down to 0).
    nproblem_solves : int
        Maximum adjoint Picard iterations for ``"strong"`` coupling.
    coupling : list
        One entry per group in ``problems``:
        ``"weak"``        – adjoint of each solver, reversed.
        ``"strong"``      – partitioned adjoint Picard iteration.
        ``"monolithic"``  – solve transposed global block adjoint system.
    parameters : dict
        Design variables / fixed problem parameters (read-only).
    context : dict
        Solver metadata; augmented with ``"tstep"`` and ``"solve_iter"``.
    """
    adj = {}
    #
    for tstep in reversed(range(ntimesteps)):
        # adjoint groups are solved in reverse order of the forward pass
        for i, problem in reversed(list(enumerate(problems))):
            #
            if coupling[i] == "weak":
                # single solver
                if isinstance(problem, ProblemSolver):
                    adj_out = problem.adjoint(rhs=adj_rhs[problem.fieldname],
                                              state=state,
                                              parameters=parameters,
                                              solver_kw=solver_kw[i],
                                              lin_solver_kw=lin_solver_kw[i],
                                              preconditioner_kw=preconditioner_kw[i],
                                              logger=logger)
                    adj.update(adj_out)
                # list of solvers: adjoint in reverse sequence
                else:
                    for solver in reversed(problem):
                        adj_out = solver.adjoint(rhs=adj_rhs[solver.fieldname],
                                                 state=state,
                                                 parameters=parameters,
                                                 solver_kw=solver_kw[i],
                                                 lin_solver_kw=lin_solver_kw[i],
                                                 preconditioner_kw=preconditioner_kw[i],
                                                 logger=logger)
                        adj.update(adj_out)
            #
            elif coupling[i] == "strong":
                # partitioned adjoint Picard iteration
                for solve_iter in range(nproblem_solves):
                    adj_old = copy_relevant_fields(adj, problem)
                    for solver in reversed(problem):
                        adj_out = solver.adjoint(rhs=adj_rhs[solver.fieldname],
                                                 state=state,
                                                 parameters=parameters,
                                                 solver_kw=solver_kw[i],
                                                 lin_solver_kw=lin_solver_kw[i],
                                                 preconditioner_kw=preconditioner_kw[i],
                                                 logger=logger)
                        adj.update(adj_out)
                    if converged(adj_old, adj):
                        break
            #
            elif coupling[i] == "monolithic":
                # collect transposed blocks from each solver
                blocks_T = {}
                rhs_adj  = {}
                for solver in problem:
                    local = solver.assemble_blocks(state, parameters,
                                                   solver_kw[i], logger)
                    blocks_T.update(transpose_blocks(local["blocks"]))
                    rhs_adj.update(local["adj_rhs"])
                system_T = assemble_global_block_system(blocks=blocks_T,
                                                        rhs=rhs_adj,
                                                        state=state,
                                                        parameters=parameters,
                                                        solver_kw=solver_kw[i],
                                                        logger=logger)
                solution = solve_global_block_system(system_T)
                adj.update(split_global_solution_by_fields(solution, problem))
            #
            else:
                raise ValueError(f"Unknown coupling mode: {coupling[i]}")
    return adj