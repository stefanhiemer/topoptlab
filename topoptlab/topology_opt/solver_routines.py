# SPDX-License-Identifier: GPL-3.0-or-later
from collections.abc import Callable
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
# logging related stuff
from topoptlab.log_utils import BaseLogger
# 
from topoptlab.problem_solver import ProblemSolver
from topoptlab.linear_solvers import res_norm
from topoptlab.block_assembly import (assemble_global_block_system,
                                      solve_global_block_system,
                                      split_global_solution_by_fields,
                                      transpose_blocks)

def copy_relevant_fields(state: dict, problem) -> dict:
    """Shallow-copy only the fields written by solvers in problem."""
    solvers = [problem] if isinstance(problem, ProblemSolver) else problem
    keys = {k for solver in solvers for k in solver.output_keys}
    return {k: state[k].copy() for k in keys if k in state}

def converged(old: dict, new: dict, atol: float = 1e-8) -> bool:
    """True when the 2-norm of the field change across all keys in old is below atol."""
    r = np.concatenate([new[k].ravel() - old[k].ravel() for k in old])
    return res_norm(r, atol=atol)

def initialize_problems(problems: list[Callable | list[Callable]],
                        bcs: Callable | list[Callable],
                        mesh_kw: dict,
                        solver_kw: list[dict],
                        logger: None | BaseLogger = None
                        ) -> list[ProblemSolver | list[ProblemSolver]]:
    """
    Instantiate each problem group's solver class(es) into ``ProblemSolver``
    instances, in place.

    Each entry of ``problems`` is either a single (uninitialized) solver
    class or a list of solver classes forming a coupled group. Every group
    is paired with its own boundary condition callable and keyword
    arguments, and instantiated as
    ``solver(**mesh_kw, 
             **solver_kw[i], 
             bc=bcs[i], 
             logger=logger)``.

    Parameters
    ----------
    problems : list of callable or list of list of callable
        One entry per problem group: either a single uninitialized solver
        class, or a list of solver classes forming a coupled group.
    bcs : callable or list of callable
        Boundary condition callable(s), one per problem group. A single
        callable is broadcast to a one-element list.
    mesh_kw : dict
        Mesh/discretization keyword arguments forwarded to every solver.
    solver_kw : list of dict
        Per-problem-group keyword arguments forwarded to the solver
        constructor(s).
    logger : BaseLogger or None
        Logger forwarded to every solver constructor.

    Returns
    -------
    problems : list of ProblemSolver or list of list of ProblemSolver
        The same list, with every entry replaced in place by its
        instantiated solver(s).
    """
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

def solver_loop(problems: list,
                state: dict,
                ntimesteps: int,
                lin_solver_kw: list[dict],
                preconditioner_kw: list[dict],
                interpol_kw: list[dict | list[dict]],
                nproblem_solves: int = 100,
                coupling: list = None,
                parameters: dict = {},
                solver_kw: list[dict] = None,
                logger: None | BaseLogger = None,
                ) -> dict:
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
    interpol_kw : list
        One entry per problem group: a ``dict`` for a single solver, or a
        ``list`` of dicts (one per solver) for coupled/monolithic groups.
        Written into ``parameters["interpol_kw"]`` before each solver call.
    """
    # TO DO: check that all lists have same length
    #
    for tstep in np.arange(ntimesteps):
        # iterate problem groups in forward order
        for i, problem in enumerate(problems):
            #
            if coupling[i] == "weak":
                # single solver or list of solvers solved once in sequence
                if isinstance(problem, ProblemSolver):
                    solvers = [problem]
                    ikw_list = [interpol_kw[i]]
                else:
                    solvers = problem
                    ikw_list = interpol_kw[i]
                for solver, ikw in zip(solvers, ikw_list):
                    parameters["interpol_kw"] = ikw
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
                    for solver, ikw in zip(problem, interpol_kw[i]):
                        parameters["interpol_kw"] = ikw
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
                for solver, ikw in zip(problem, interpol_kw[i]):
                    parameters["interpol_kw"] = ikw
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

def adjoint_loop(problems: list,
                 state: dict,
                 adj_rhs: dict,
                 ntimesteps: int,
                 nproblem_solves: int = 100,
                 coupling: None | str | list[str] = None,
                 parameters: dict = {},
                 solver_kw: list[dict] = None,
                 lin_solver_kw: list[dict] = None,
                 preconditioner_kw: list[dict] = None,
                 interpol_kw: list[dict | list[dict]] = None,
                 logger = None,
                 ) -> dict:
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
    interpol_kw : list
        One entry per problem group: a ``dict`` for a single solver, or a
        ``list`` of dicts (one per solver) for coupled/monolithic groups.
        Written into ``parameters["interpol_kw"]`` before each solver call.
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
                    parameters["interpol_kw"] = interpol_kw[i]
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
                    for solver, ikw in zip(reversed(problem), reversed(interpol_kw[i])):
                        parameters["interpol_kw"] = ikw
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
                    for solver, ikw in zip(reversed(problem), reversed(interpol_kw[i])):
                        parameters["interpol_kw"] = ikw
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
                for solver, ikw in zip(problem, interpol_kw[i]):
                    parameters["interpol_kw"] = ikw
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