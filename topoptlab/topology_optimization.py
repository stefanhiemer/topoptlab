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
from topoptlab.linear_solvers import res_norm
from topoptlab.block_assembly import (assemble_global_block_system,
                                      solve_global_block_system,
                                      split_global_solution_by_fields,
                                      transpose_blocks)

def update_history(xhist: List, x: np.ndarray,
                   xPhys_hist: Union[None, List], xPhys: np.ndarray,
                   obj_hist: Union[None, List], obj: float,
                   constrs_hist: Union[None, List], constrs: np.ndarray,
                   max_history: int) -> None:
    """
    Append current iterate to history lists and prune to max_history+1 entries.

    Parameters
    ----------
    xhist : list of np.ndarray
        history of design iterates.
    x : np.ndarray
        current design iterate.
    xPhys_hist : None or list of np.ndarray
        history of physical densities, or None if not tracked.
    xPhys : np.ndarray
        current physical densities.
    obj_hist : None or list of float
        history of objective values, or None if not tracked.
    obj : float
        current objective value.
    constrs_hist : None or list of np.ndarray
        history of constraint vectors, or None if not tracked.
    constrs : np.ndarray
        current constraint vector.
    max_history : int
        maximum number of iterates to retain.

    Returns
    -------
    None
    """
    # append history
    xhist.append(x.copy())
    if xPhys_hist is not None:
        xPhys_hist.append(xPhys.copy())
    if obj_hist is not None:
        obj_hist.append(obj)
    if constrs_hist is not None:
        constrs_hist.append(constrs.copy())
    # prune history
    if len(xhist) > max_history+1:
        del xhist[:len(xhist)-max_history-1]
    if xPhys_hist is not None and len(xPhys_hist) > max_history+1:
        del xPhys_hist[:len(xPhys_hist)-max_history-1]
    if obj_hist is not None and len(obj_hist) > max_history+1:
        del obj_hist[:len(obj_hist)-max_history-1]
    if constrs_hist is not None and len(constrs_hist) > max_history+1:
        del constrs_hist[:len(constrs_hist)-max_history-1]
    return

def prepare_phys_problems(mesh: dict,
                          lk: Union[None, Callable],
                          body_forces_kw: dict,
                          assembly_mode: str,
                          obj_kw: dict) -> Tuple:
    """
    Build all static FEM data needed before the optimisation loop.

    Parameters
    ----------
    mesh : dict
        Mesh dictionary with keys ``"nelx"``, ``"nely"``, ``"nelz"``,
        ``"ndim"``, ``"l"``.
    lk : callable or None
        Element stiffness function ``KE = lk(l=l)``.  Defaults to standard
        linear-elasticity stiffness for the spatial dimension in ``mesh``.
    body_forces_kw : dict
        Body-force specification.  Supported keys:

        ``"strain_uniform"``
            Uniform applied strain in Voigt notation, shape ``(n_strain,)``
            or ``(n_strain, n_lc)``.  1-D arrays are promoted to 2-D
            in-place.  The single-element reference displacement ``u0``
            is written into ``obj_kw["u0"]`` (homogenisation).
        ``"density_coupled"``
            Density-dependent body-force vector ``b``.
    assembly_mode : str
        ``"full"`` or ``"lower"`` (exploit symmetry of K).
    obj_kw : dict
        Keyword arguments for the objective function.  Updated with ``u0``
        when ``"strain_uniform"`` body forces are present.

    Returns
    -------
    lk : callable
        Resolved element stiffness function.
    KE : np.ndarray, shape (dof_el, dof_el)
        Reference element stiffness matrix.
    n_nodaldof : int
        Number of DOFs per node.
    ndof : int
        Total number of DOFs.
    edofMat : np.ndarray, shape (n_el, dof_el)
        Element DOF connectivity matrix.
    iK, jK : np.ndarray
        Row / column indices for sparse global stiffness assembly.
    assm_indcs : np.ndarray or None
        Lower-triangle index pairs used when ``assembly_mode == "lower"``.
    fe_strain : np.ndarray or None
        Pre-computed strain body-force vectors, shape ``(dof_el, n_lc)``.
    fe_dens : np.ndarray or None
        Pre-computed density body-force vector.
    """
    ndim = mesh["ndim"]
    nelx, nely, nelz = mesh["nelx"], mesh["nely"], mesh["nelz"]
    l = mesh["l"]
    if ndim == 2:
        create_edofMat = create_edofMat2d
    else:
        create_edofMat = create_edofMat3d
    # get function of element stiffness matrix
    if lk is None and ndim == 2:
        lk = lk_linear_elast_2d
    elif lk is None and ndim == 3:
        lk = lk_linear_elast_3d


    #
    # get element stiffness matrix
    KE = lk(l=l)
    # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3 D
    n_nodaldof = int(KE.shape[-1]/2**ndim)
    # total number of nodal dofs
    ndof = n_nodaldof * np.prod( np.array([nelx,nely,nelz][:ndim])+1 )
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,nely=nely,nelz=nelz,
                                             nnode_dof=n_nodaldof)
    # fetch body forces
    if len(body_forces_kw.keys())==0:
        fe_strain = None
        fe_dens = None
    else:
        # assume each strain is a column vector in Voigt notation
        if "strain_uniform" in body_forces_kw.keys():
            # fetch functions to create body force
            if ndim == 2:
                lf = lf_strain_2d
            elif ndim == 3:
                lf = lf_strain_3d
            # calculate forces for each strain
            fe_strain = []
            if len(body_forces_kw["strain_uniform"].shape) == 1:
                body_forces_kw["strain_uniform"] = body_forces_kw["strain_uniform"][:,None]
            #
            for i in range(body_forces_kw["strain_uniform"].shape[-1]):
                fe_strain.append(lf(body_forces_kw["strain_uniform"][:,i],E=1.0, l=l))
            fe_strain = np.column_stack(fe_strain)
            # find the imposed elemental field. Material properties are
            # unimportant here as it just depends on the geometry of the
            # element, not its properties. This part is needed for
            # homogenization related objective functions and may later
            # become optional via some flags.
            if ndim == 2 and n_nodaldof != 1:
                fixed = np.array([0,1,3])
            elif ndim == 3 and n_nodaldof != 1:
                fixed = np.array([0,1,2,4,5,7,8])
            elif n_nodaldof == 1:
                fixed = np.array([0])
            free = np.setdiff1d(np.arange(KE.shape[-1]), fixed)
            u0 = np.zeros(fe_strain.shape)
            u0[free] = np.linalg.solve(KE[free,:][:,free],
                                       fe_strain[free,:])
            if "u0" not in obj_kw.keys():
                obj_kw["u0"] = u0
        else:
            fe_strain = None
        #
        if "density_coupled" in body_forces_kw.keys():
            # fetch functions to create body force
            if ndim == 2 and n_nodaldof!=1:
                lf = lf_bodyforce_2d
            elif ndim == 3 and n_nodaldof!=1:
                lf = lf_bodyforce_3d
            fe_dens = lf_bodyforce_2d(b=body_forces_kw["density_coupled"])
        else:
            fe_dens = None
        #
        if len([key for key in body_forces_kw.keys() \
                if key not in ["density_coupled","strain_uniform"]]):
            raise NotImplementedError("One type of bodyforce/source has not yet been implemented.")
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat, 
                              mode=assembly_mode)
    if assembly_mode == "lower":
        assm_indcs = np.column_stack(np.tril_indices_from(KE))
        assm_indcs = assm_indcs[np.lexsort( (assm_indcs[:,0],assm_indcs[:,1]) )]
    else:
        assm_indcs = None

    return lk, KE, n_nodaldof, ndof, edofMat, iK, jK, assm_indcs, fe_strain, fe_dens

def solve_phys_problems(xPhys: np.ndarray,
                        KE: np.ndarray,
                        iK: np.ndarray,
                        jK: np.ndarray,
                        ndof: int,
                        n: int,
                        edofMat: np.ndarray,
                        f: np.ndarray,
                        fixed: Union[np.ndarray, List[np.ndarray]],
                        free: Union[np.ndarray, List[np.ndarray]],
                        springs: Any,
                        matinterpol: Callable,
                        matinterpol_kw: Dict,
                        assembly_mode: str,
                        assm_indcs: Union[None, np.ndarray],
                        body_forces_kw: Dict,
                        fe_strain: Union[None, np.ndarray],
                        fe_dens: Union[None, np.ndarray],
                        lin_solver: str,
                        lin_solver_kw: Dict,
                        preconditioner: Union[None, str],
                        preconditioner_kw: Dict,
                        u: np.ndarray,
                        ) -> Tuple[np.ndarray, Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble and solve all physical (FE) problems for the current iterate. Does the following steps:

    1. computes scaled element stiffness matrices based on physical variables (e. g. xPhys),
    2. assembles the global stiffness matrix 
    3. adds design dependent force contributions (e. g. body-forces) to the right-hand side
    4. applies boundary conditions 
    5. solves the linear system for all load cases simultaneously 
    
    The returned objects are everything needed by the subsequent adjoint analysis and sensitivity computation.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n, k)
        Physical element densities.
    KE : np.ndarray, shape (dof_el, dof_el)
        Reference element stiffness matrix (unscaled).
    iK : np.ndarray
        Row indices for sparse global stiffness assembly.
    jK : np.ndarray
        Column indices for sparse global stiffness assembly.
    ndof : int
        Total number of degrees of freedom.
    n : int
        Total number of elements.
    edofMat : np.ndarray, shape (n, dof_el)
        Element degree-of-freedom connectivity matrix.
    f : np.ndarray, shape (ndof, n_lc)
        External nodal force array for all load cases.
    fixed : np.ndarray or list of np.ndarray
        Indices of fixed (Dirichlet) DOFs. A single array is shared across
        all load cases; a list provides per-load-case DOFs (future extension).
    free : np.ndarray or list of np.ndarray
        Indices of free DOFs, complementary to ``fixed``.
    springs : Any
        Spring data passed to :func:`assemble_matrix`. ``None`` if unused.
    matinterpol : callable
        Material interpolation ``scale = matinterpol(xPhys, **matinterpol_kw)``.
    matinterpol_kw : dict
        Keyword arguments forwarded to ``matinterpol``.
    assembly_mode : str
        Stiffness assembly mode. ``"full"`` uses the full element matrix;
        ``"lower"`` exploits symmetry and only uses half of the off-diagonal.
    assm_indcs : np.ndarray or None
        Lower-triangular index pairs used when ``assembly_mode == "lower"``.
    body_forces_kw : dict
        Body-force specification. Supported keys: ``"strain_uniform"`` and
        ``"density_coupled"``.
    fe_strain : np.ndarray or None
        Element load vectors for uniform strain body forces, shape
        ``(n, dof_el, n_lc)``. Required when ``"strain_uniform"`` is in
        ``body_forces_kw``.
    fe_dens : np.ndarray or None
        Element load vectors for density-coupled body forces, shape
        ``(n, dof_el, n_lc)``. Required when ``"density_coupled"`` is in
        ``body_forces_kw``.
    lin_solver : str
        Name of the linear solver backend.
    lin_solver_kw : dict
        Additional keyword arguments for the linear solver.
    preconditioner : str or None
        Name of the preconditioner, or ``None`` for no preconditioning.
    preconditioner_kw : dict
        Additional keyword arguments for the preconditioner.
    u : np.ndarray, shape (ndof, n_lc)
        Displacement field used as the initial guess; updated in-place.

    Returns
    -------
    u : np.ndarray, shape (ndof, n_lc)
        Updated displacement field for all load cases.
    K : Any
        Assembled, BC-reduced stiffness matrix (solver-specific format).
        Reused for all adjoint solves within the same outer iteration.
    fact : Any
        Factorization of ``K`` returned by the linear solver. Passed back
        to :func:`solve_lin` via ``factorization=fact`` for efficient adjoint
        solves without re-factorization.
    precond : Any
        Preconditioner object, or ``None`` if no preconditioner was used.
    Kes : np.ndarray, shape (n, dof_el, dof_el)
        Scaled element stiffness matrices ``KE * scale``. Required by
        objective and constraint functions.
    rhs : np.ndarray, shape (ndof, n_lc)
        Full right-hand side ``f + f_body`` before boundary conditions.
    f_body : np.ndarray, shape (ndof, n_lc)
        Body-force contribution to the right-hand side. Needed separately
        for density-coupled body-force sensitivity terms.
    """
    # update physical properties of the elements and thus the entries
    # of the elements
    scale = matinterpol(xPhys=xPhys,**matinterpol_kw)
    Kes = KE[None,:,:]*scale[:,:,None]
    if assembly_mode == "full":
        # this here is more memory efficient than Kes.flatten() as it
        # provides a view onto the original Kes array instead of a copy
        sK = Kes.reshape(np.prod(Kes.shape))
    elif assembly_mode == "lower":
        sK = Kes[:,
                 assm_indcs[:,0],
                 assm_indcs[:,1]].reshape(n*int(KE.shape[-1]/2*(KE.shape[-1]+1)))
    # setup and solve FE problem
    # assemble system matrix
    K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                        ndof=ndof,solver=lin_solver,
                        springs=springs)
    # assemble forces due to body forces
    f_body = np.zeros(f.shape)
    u0 = None
    for bodyforce in body_forces_kw.keys():
        # assume each strain is a column vector in Voigt notation
        if "strain_uniform" in body_forces_kw.keys():
            fes = fe_strain[None,:,:]*scale[:,:,None]
            np.add.at(f_body,
                        edofMat,
                        fes)
        if "density_coupled" in body_forces_kw.keys():
            fes = fe_dens[None,:,:]*simp(xPhys=xPhys, eps=0., penal=1.)[:,:,None]
            np.add.at(f_body,
                        edofMat,
                        fes)
    # assemble right hand side
    rhs = f+f_body
    # apply boundary conditions to matrix
    K = apply_bc(K=K,solver=lin_solver,
                    free=free,fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[free, :], fact, precond = solve_lin(K=K, 
                                    rhs=rhs[free],
                                    rhs0=u[free,:],
                                    solver=lin_solver,
                                    solver_kw=lin_solver_kw,
                                    preconditioner=preconditioner,
                                    preconditioner_kw=preconditioner_kw)
    return u, K, fact, precond, Kes, rhs, f_body

def copy_relevant_fields(state: Dict, problem) -> Dict:
    """Shallow-copy only the fields written by solvers in problem."""
    solvers = [problem] if isinstance(problem, ProblemSolver) else problem
    keys = {k for solver in solvers for k in solver.output_keys}
    return {k: state[k].copy() for k in keys if k in state}


def converged(old: Dict, new: Dict, atol: float = 1e-8) -> bool:
    """True when the 2-norm of the field change across all keys in old is below atol."""
    r = np.concatenate([new[k].ravel() - old[k].ravel() for k in old])
    return res_norm(r, atol=atol)


def solver_loop(problems: List,
                state: Dict,
                ntimesteps: int,
                nproblem_solves: int = 100,
                coupling: List = None,
                parameters: Dict = {},
                solver_kw: List[Dict] = None,
                logger = None,
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
                                                 logger)
                        out    = solver.solve(system, 
                                              state, 
                                              parameters, 
                                              solver_kw[i], 
                                              logger)
                    else:
                        out = solver.solve({}, 
                                           state, 
                                           parameters, 
                                           solver_kw[i], 
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
                                                     logger)
                            out    = solver.solve(system, 
                                                  state, 
                                                  parameters, 
                                                  solver_kw[i], 
                                                  logger)
                        else:
                            out = solver.solve({}, 
                                               state, 
                                               parameters, 
                                               solver_kw[i], 
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
                    adj_out = problem.adjoint(rhs=adj_rhs,
                                              state=state,
                                              parameters=parameters,
                                              solver_kw=solver_kw[i],
                                              logger=logger)
                    adj.update(adj_out)
                # list of solvers: adjoint in reverse sequence
                else:
                    for solver in reversed(problem):
                        adj_out = solver.adjoint(rhs=adj_rhs,
                                                 state=state,
                                                 parameters=parameters,
                                                 solver_kw=solver_kw[i],
                                                 logger=logger)
                        adj.update(adj_out)
            #
            elif coupling[i] == "strong":
                # partitioned adjoint Picard iteration
                for solve_iter in range(nproblem_solves):
                    adj_old = copy_relevant_fields(adj, problem)
                    for solver in reversed(problem):
                        adj_out = solver.adjoint(rhs=adj_rhs,
                                                 state=state,
                                                 parameters=parameters,
                                                 solver_kw=solver_kw[i],
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
                    blocks_T.update(transpose_blocks(local.get("blocks", {})))
                    rhs_adj.update(local.get("adj_rhs", {}))
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

def initialize_design(n: int,
                      initial_guess: Union[None, Dict[str, np.ndarray]],
                      volfrac: Union[None, float],
                      n_mat: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize design variables x and physical densities xPhys.

    Parameters
    ----------
    n : int
        Total number of design elements.
    initial_guess : dict or None
        Optional initial values.  Recognised keys:

        ``"x"``
            Initial design variables, shape (n, n_mat).  Defaults to uniform
            ``volfrac`` (or 0.5 if ``volfrac`` is None).
        ``"xPhys"``
            Initial physical densities, shape (n, n_mat).  Defaults to a copy
            of ``x``.

    volfrac : float or None
        Volume fraction used to fill ``x`` when no initial guess is given.
        If None, defaults to 0.5.
    n_mat : int
        Number of materials; determines the second axis of x and xPhys.

    Returns
    -------
    x : np.ndarray, shape (n, n_mat)
        Design variables.
    xPhys : np.ndarray, shape (n, n_mat)
        Physical densities.
    """
    if initial_guess is None or "x" not in initial_guess:
        x = np.full(shape=(n, n_mat),
                    fill_value=volfrac if volfrac is not None else 0.5,
                    dtype=float,
                    order='F')
    else:
        x = initial_guess["x"]
    if initial_guess is None or "xPhys" not in initial_guess:
        xPhys = x.copy()
    else:
        xPhys = initial_guess["xPhys"]
    return x, xPhys

# MAIN DRIVER
def main(nelx: int, nely: int,
         volfrac: float, #penal: float, 
         rmin: float, 
         ft: [int,TOFilter,List[TOFilter]] = 1,
         filter_kw: Union[Dict,List] = {},
         simulation_kw: Dict = {"grid": "regular",
                                "element order": 1,
                                "meshfile": None},
         nelz: Union[None,int] = None,
         initial_guess: Union[None,Dict[str, np.ndarray]] = None,
         filter_mode: str = "convolution",
         lin_solver_kw: Dict = {"name": "scipy-direct"}, 
         preconditioner_kw: Dict = {"name": None},
         assembly_mode: str = "full",
         materials_kw: Dict = {"E": 1.}, 
         body_forces_kw: Dict = {},
         bcs: Callable = mbb_2d,
         problems: Union[None, List[ProblemSolver]] = None,
         solver_kw: Union[Dict, List[Dict]] = {},
         solver_coupling: Union[None, List[str]] = None,
         lk: Union[None,Callable] = None,
         l: Union[float,List,np.ndarray] = 1.,
         obj_func: Callable = compliance, 
         obj_kw: Dict = {},
         matinterpol: Callable = simp,
         matinterpol_dx: Callable = simp_dx,
         matinterpol_kw: Dict = {"eps": 1e-9, "penal": 3.},
         el_flags: Union[None,np.ndarray] = None,
         el_flags_policy: Union[None,Dict] = None,
         constraints: Union[None,List[Dict]] = None,
         optimizer: str = "mma",
         optimizer_kw: Union[None,Dict] = None,
         mix: Union[None,float] = None,
         accelerator_kw: Dict = {"accel_freq": 4,
                                 "accel_start": 20,
                                 "max_history": 0,
                                 "accelerator": None},
         convergence_kw: Dict = {"conv_tol": 1e-2,
                                 "change_func": max_design_change},
         continuation_kw: Union[None, Dict] = None,
         nouteriter: int = 2000, 
         ninneriter: int = 15,
         mesh_file: Union[None,str] = None,
         output_kw: Dict = default_outputkw()) -> Tuple[np.ndarray,float]:
    """
    Topology optimization workflow with the material interpolation method. 
    Can treat single physics stationary problems.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    volfrac : float
        volume fraction.
    penal : float
        penalty exponent for the SIMP method.
    rmin : float
        cutoff radius for the filter. Only elements within the element-center
        to element center distance are used for filtering.
    ft : int or TOFilter or list
        integer flag for the filter. 0 sensitivity filtering,
        1 density filtering, -1 no filter.
    nelz : int or None
        number of elements in z direction. If None, simulation is 2d.
    initial_guess : None or dict
        dictionary with initial values for design or intermediate variables. 
        Supported keys are "x", " xTilde-number" and "xPhys". The value for 
        "x" initializes the design variables, "xPhys" initializes the physical 
        densities, and any key of the form " xTilde-number" initializes the 
        matching entry in the xTilde list when a list of filters is used.
    filter_mode : str
        indicates how filtering is done. Possible values are "matrix" or
        "helmholtz". If "matrix", then density/sensitivity filters are
        implemented via a sparse matrix and applied by multiplying
        said matrix with the densities/sensitivities.
    assembly_mode : str
        whether full or only lower triangle of linear system / matrix is
        created.
    materials_kw : dict
        dictionary containing all materials and their properties. Conventions 
        must still be determined.
    bcs : str or callable
        returns the boundary conditions
    lk : None or callable
        element stiffness matrix
    l : float or tuple of length (ndim) or np.ndarray of shape (ndim)
        side lengths of each element
    obj_func : callable
        objective function. Should update the objective value, the rhs of the
        the adjoint problem (currently only for stationary lin. problems) and
        a flag indicating whether the objective is self adjoint.
    obj_kw : dict
        keywords needed for the objective function. E. g. for a compliant
        mechanism and maximization of the displacement it would be the
        indicator array for output nodes. Check the objective for the necessary
        entries.
    matinterpol : callable 
        callable for material interpolation. Default is SIMP (simp).
    matinterpol_dx : callable 
        callable of derivative of the material interpolation with regards to 
        the design variable. Default is SIMP (simp_dx).
    matinterpol_kw : callable 
        dictionary containing the arguments for the material interpolation.
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements:
        0 = free (optimised normally), 1 = passive (fixed at 0),
        2 = active (fixed at 1), 3 = non-design (excluded from the optimizer
        but participates in the density filter — its physical density is
        determined by the filter from surrounding elements, and its gradient
        contribution propagates to neighbours before being zeroed).
    optimizer: str
        solver options which are "oc", "mma" and "gcmma" for the optimality
        criteria method, the method of moving asymptotes and the globally
        covergent method of moving asymptotes.
    optimizer_kw : dict
        dictionary with parameters for optimizer.
    mix : None or float,
        mixing parameter for design variable update.
    nouteriter: int
        number of TO iterations
    ninneriter : int
        number of inner iterations for GCMMA.
    convergence_kw : dict
        dictionary controlling convergence. Supported keys are:

        - ``conv_tol`` (float): tolerance on the change metric below which
          convergence is declared. Default 1e-1.
        - ``change_func`` (callable or None): function with signature
          ``change_func(x, xhist) -> float`` that computes the scalar change
          metric each iteration. If None, the default
          ``np.abs(xhist[-1] - xhist[-2]).max()`` (inf-norm of the update) is
          used.
    continuation_kw : None or dict
        dictionary controlling parameter continuation between stages. If None,
        the loop terminates as soon as ``convergence_kw["conv_tol"]`` is
        reached. Otherwise must contain two parallel lists:

        - ``"funcs"`` : list of callables, each with signature
          ``f(hist, change, loop, filter_kw, conv_tol, continuation_kw) -> bool``.
        - ``"func_kws"`` : list of dicts, one per function, holding the
          private keyword arguments for that function. Each dict is passed
          as ``continuation_kw`` to the corresponding callable and may be
          mutated in-place to maintain per-function state across iterations.

        The loop continues as long as at least one function returns True;
        it terminates only when all return False.
    output_kw : dict
        dictionary controlling output and logging. Missing keys are filled
        from ``default_outputkw()``. Recognised keys:

        - ``"file"``         : str  — base name for log and VTK output files
          (default ``"topopt"``).
        - ``"display"``      : bool — if True, the physical density field is
          plotted to screen each iteration (default True).
        - ``"export"``       : bool — if True, the final design is exported to
          a VTK file via ``export_vtk`` (default True).
        - ``"write_log"``    : bool — if True, a ``SimpleLogger`` writing to
          ``<file>.log`` is created; otherwise an ``EmptyLogger`` is used
          (default True).
        - ``"verbosity"``    : int  — verbosity level passed to
          ``SimpleLogger``; higher values produce more output (default 20).
        - ``"profile"``      : bool — if True, ``cProfile`` is enabled for
          the optimisation loop and results are printed on exit (default
          False).
        - ``"output_movie"`` : bool — if True, each iteration is exported as
          a separate VTK file named ``<file>_<iter>.vtk``, suitable for
          assembling a movie (default False).

    Returns
    -------
    None.

    """
    # insert: checking_function
    # check dictionaries
    check_output_kw(output_kw)
    check_simulation_params(simulation_kw)
    if el_flags_policy is not None:
        check_el_flags_policy(el_flags_policy)
    # initialize profiling
    if output_kw["profile"]:
        profiler = Profile() 
        profiler.enable()
    # extract linear solver and preconditioner
    lin_solver = lin_solver_kw["name"]
    preconditioner = preconditioner_kw["name"]
    lin_solver_kw = dict_without(lin_solver_kw, "name")
    preconditioner_kw = dict_without(preconditioner_kw, "name")
    # normalize materials_kw: single dict → one-element list
    if isinstance(materials_kw, dict):
        materials_kw = [materials_kw]
    n_mat = len(materials_kw)
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if output_kw["write_log"]:
        # check if log file exists and if True delete
        log = SimpleLogger(file=output_kw["file"],
                           verbosity=output_kw["verbosity"])
        
        #
        log.info(f"date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log.info(f"optimizer {optimizer}")
        log.info(f"number of spatial dimensions: {ndim}")
        log.info("elements: "+" x ".join([f"{nelx}",f"{nely}",f"{nelz}"][:ndim]))
        if volfrac is not None:
            log.info(f"volfrac: {volfrac} rmin: {rmin}") # penal: {penal}")
        else:
            log.info(f"rmin: {rmin}")#  penal: {penal}")
        if isinstance(ft, int):
            log.info("filter: " + ["Sensitivity based",
                                  "Density based",
                                  "Haeviside Guest",
                                  "Haeviside complement Sigmund 2007",
                                  "Haeviside eta projection",
                                  "Volume Preserving eta projection",
                                  "No filter"][ft])
        else:
            log.info("filter: custom")
        log.info(f"filter mode: {filter_mode}")
    else:
        # check if log file exists and if True delete
        log = EmptyLogger()
    # total number of design elements
    n_el = int(np.prod([nelx, nely, nelz][:ndim]))
    #
    if isinstance(l, float):
        l = np.array([l for i in np.arange(ndim)])
    # bundle all mesh-related information; irregular-mesh support will extend this
    if mesh_file is None:
        if ndim == 2:
            _mapping = partial(map_eltoimg,  nelx=nelx, nely=nely)
            _invmapping = partial(map_imgtoel,  nelx=nelx, nely=nely)
        else:
            _mapping = partial(map_eltovoxel, nelx=nelx, nely=nely, nelz=nelz)
            _invmapping = partial(map_voxeltoel, nelx=nelx, nely=nely, nelz=nelz)
        mesh = {"nelx": nelx,
                "nely": nely,
                "nelz": nelz,
                "ndim": ndim,
                "l": l,
                "n_el": n_el,
                "cellVolume": float(np.prod(l) * n_el),
                "mapping": _mapping,
                "invmapping": _invmapping}
    else:
        raise NotImplementedError("Cannot handle irregular meshes right now.")
    mapping, invmapping = mesh["mapping"], mesh["invmapping"]
    # Allocate design variables (as array), initialize and allocate sens.
    x, xPhys = initialize_design(n=n_el, initial_guess=initial_guess, volfrac=volfrac, n_mat=n_mat)
    # precompute prescribed-element masks for filter policy corrections
    passive_mask, active_mask, prescribed_mask = None, None, None
    if el_flags is not None:
        passive_mask = el_flags == 1
        active_mask = el_flags == 2
        prescribed_mask = (el_flags == 1) | (el_flags == 2) 
    # initialize arrays for gradients
    dobj = np.zeros( x.shape,order="F")
    # build and validate constraint list
    if constraints is None:
        constraints = []
    # legacy: if volfrac is given, prepend the volume fraction inequality
    if volfrac is not None:
        constraints = [{"name": "volume fraction",
                        "func": volume_fraction_constraint,
                        "type": "leq",
                        "value": volfrac,
                        "kw":   {}}] + list(constraints)
        if ft == 0:
            constraints[0]["filter"] = False
    # fill missing optional keys and check types
    check_constraints(constraints)
    # MMA/GCMMA only handle inequalities: split equality constraints into two
    # inequalities upfront so the rest of the pipeline only ever sees leq/geq
    if optimizer in ["mma", "gcmma"]:
        constraints = expand_eq_constraints(constraints)
    n_constr = len(constraints)
    # build filter mask: one bool per constraint row
    _constr_filter_mask = np.array([c["filter"] for c in constraints], dtype=bool)
    constrs  = np.zeros( (n_constr, 1) )
    dconstrs = np.zeros((n_el, n_constr))
    # initialize history length needed for optimizer
    optimizer_kw = check_optimizer_kw(optimizer=optimizer,
                                      n=x.shape[0],
                                      ft=ft,
                                      n_constr=n_constr,
                                      optimizer_kw=optimizer_kw)
    if optimizer in ["oc","ocm"]:
        # legacy OC: only a single constraint (volume fraction) is supported
        if n_constr != 1:
            raise ValueError(f"Optimizers 'oc' and 'ocm' support exactly one constraint, "
                             f"got {n_constr}.")
        # must be initialized to use the NGuyen/Paulino OC approach
        g = 0
        max_history = 2
    elif optimizer == "ocg":
        # must be initialized to use the NGuyen/Paulino OC approach
        g = 0
        max_history = 2
    elif optimizer in ["mma","gcmma"]:
        # mma needs results of the two previous iterations
        max_history = 3 
        # handle element element flags
        if el_flags is not None:
            # passive
            mask = el_flags == 1
            optimizer_kw["xmin"][mask] = 0.
            optimizer_kw["xmax"][mask] = 0.+1e-9
            x[mask,0] = 0.
            xPhys[mask,0] = 0.
            # active
            mask = el_flags == 2
            optimizer_kw["xmin"][mask] = 1.- 1e-9
            optimizer_kw["xmax"][mask] = 1.
            x[mask, :] = 1.
            xPhys[mask, :] = 1.
            # redistribute volfrac over free elements so that passive elements
            # (contributing 0) add material to the free set
            if volfrac is not None:
                x_free = np.clip(volfrac * n_el / (~prescribed_mask).sum(), 0., 1.)
                x[~prescribed_mask, 0] = x_free
                xPhys[~prescribed_mask, 0] = x_free
    else:
        raise ValueError("Unknown optimizer: ", optimizer)
    # prepare FEM data structures
    if problems is None:
        lk, KE, n_nodaldof, ndof, edofMat, iK, jK, assm_indcs, fe_strain, fe_dens = \
            prepare_phys_problems(mesh=mesh, lk=lk, body_forces_kw=body_forces_kw,
                                  assembly_mode=assembly_mode, obj_kw=obj_kw)
        # set up boundary conditions
        u, f, fixed, free, springs = bcs(nelx=nelx, nely=nely, nelz=nelz, ndof=ndof)
    else:
        # mesh/DOF info pulled from the first solver — all setup was done at construction
        _solver0    = problems[0] if isinstance(problems[0], ProblemSolver) \
                      else problems[0][0]
        f = _solver0.f        # load vector shape drives adj and load-case loop
        free = _solver0.free     # needed for self-adjoint branch
        KE = _solver0._KE0    # forwarded to obj_func
        edofMat = _solver0.edofMat  # forwarded to obj_func
        u = np.zeros_like(f) # placeholder; overwritten from state each iteration
        if solver_coupling is None:
            solver_coupling = ["weak"] * len(problems)
        if isinstance(solver_kw, dict):
            solver_kw = [solver_kw] * len(problems)
        state = {}
        parameters = {"xPhys": xPhys}
        fe_strain = None
        fe_dens = None
    # initialize filters
    ft = prepare_filters(ft=ft,
                         nelx=nelx,
                         nely=nely,
                         nelz=nelz,
                         filter_mode=filter_mode,
                         rmin=rmin,
                         n_constr=n_constr,
                         l=l,
                         el_flags=el_flags,
                         el_flags_policy=el_flags_policy,
                         filter_kw=filter_kw,
                         mapping=mapping,
                         invmapping=invmapping,
                         constraint_filter_mask=_constr_filter_mask)
    # create intermediate filter variables. 
    if isinstance(ft, list):
        xTilde = []
        for i in range(len(ft) - 1):
            key = f" xTilde-{i}"
            if initial_guess is None or key not in initial_guess:
                xTilde.append(x.copy())
            else:
                xTilde.append(initial_guess[key])
    else:
        xTilde = None
    # check that boundary conditions and body forces are compatible
    f0 = None
    if fe_strain is not None:
        if f.shape[-1] != fe_strain.shape[-1]:
            raise ValueError("Number of applied strains and boundary conditions is incompatible. Last dimension must be equal.")
    if fe_dens is not None:
        if f.shape[-1] != fe_dens.shape[-1]:
            raise ValueError("Number of density based body forces and boundary conditions is incompatible. Last dimension must be equal.")
    # initialize display functions
    if output_kw["display"]:
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        if ndim == 2:
            fig,ax = plt.subplots(1,1)
            im = ax.imshow(mapping(-xPhys), cmap='gray',
                           interpolation='none', norm=Normalize(vmin=-1, vmax=0))
            plotfunc = im.set_array
        elif ndim == 3:
            raise NotImplementedError("Plotting in 3D not implemented.")
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        ax.axis("off")
        fig.show()
    #
    if output_kw["output_movie"]:
        output_kw["mov_ndigits"] = len(str(nouteriter))
    # initialize iteration history by copying initial guesses
    max_history = int(np.maximum(max_history,
                                 accelerator_kw.get("max_history", 0)))
    # check if history of xPhys is needed
    _cont_params = [inspect.signature(f).parameters
                    for f in (continuation_kw["funcs"]
                              if continuation_kw is not None else [])]
    _need_xPhys_hist = any("xPhys_hist" in p for p in _cont_params)
    if not _need_xPhys_hist and "mode" in convergence_kw.keys():
        _need_xPhys_hist = convergence_kw["mode"] == "xPhys"
    if continuation_kw is not None:
        continuation_kw["stop_flag"] = [False] * len(continuation_kw["funcs"])
    hist = {"xhist":        [x.copy() for i in np.arange(max_history)],
            "xPhys_hist":   [xPhys.copy() for i in np.arange(max_history)]
                            if _need_xPhys_hist else None,
            "obj_hist":     [0. for i in np.arange(max_history)],
            "constrs_hist": [constrs.copy() for i in np.arange(max_history)]}
    # initialize adjoint variables
    adj = np.zeros(f.shape)
    # seed filter_kw with any initial filter state (e.g. eta from EtaProjectorXu2010)
    if isinstance(ft, list):
        for _f in ft:
            if _f.changes_filter_kw:
                _f.update_filter_kw(filter_kw)
    # optimization loop
    for loop in np.arange(nouteriter):
        # solve FEM, calculate obj. func. and gradients.
        if optimizer in ["oc","mma","ocm","ocg","gcmma"]:
            ### solve physical problems
            if problems is None:
                u, K, fact, precond, Kes, rhs, f_body = solve_phys_problems(xPhys=xPhys,
                                                                            KE=KE,
                                                                            iK=iK,
                                                                            jK=jK,
                                                                            ndof=ndof,
                                                                            n=n_el,
                                                                            edofMat=edofMat,
                                                                            f=f,
                                                                            fixed=fixed,
                                                                            free=free,
                                                                            springs=springs,
                                                                            matinterpol=matinterpol,
                                                                            matinterpol_kw=matinterpol_kw,
                                                                            assembly_mode=assembly_mode,
                                                                            assm_indcs=assm_indcs,
                                                                            body_forces_kw=body_forces_kw,
                                                                            fe_strain=fe_strain,
                                                                            fe_dens=fe_dens,
                                                                            lin_solver=lin_solver,
                                                                            lin_solver_kw=lin_solver_kw,
                                                                            preconditioner=preconditioner,
                                                                            preconditioner_kw=preconditioner_kw,
                                                                            u=u)
            else:
                parameters["xPhys"] = xPhys
                state = solver_loop(problems,
                                    state,
                                    ntimesteps=1,
                                    coupling=solver_coupling,
                                    parameters=parameters,
                                    solver_kw=solver_kw,
                                    logger=log)
                u    = state[_solver0.fieldname]
                Kes  = _solver0._terms["Kes"]
            #
            for i in range(f.shape[1]): 
                log.debug("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                        loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
            # objective and sensitivities with regards to object
            obj = 0
            dobj[:] = 0.
            for i in np.arange(f.shape[1]):
                # obj. value, selfadjoint variables, self adjoint flag
                obj,rhs_adj,self_adj = obj_func(obj=obj, 
                                                i=i,
                                                xPhys=xPhys,
                                                u=u,
                                                KE=KE, 
                                                edofMat=edofMat,
                                                Kes=Kes,
                                                matinterpol=matinterpol,
                                                matinterpol_kw=matinterpol_kw,
                                                mapping=mapping, 
                                                invmapping=invmapping,
                                                cellVolume=mesh["cellVolume"],
                                                **obj_kw)
                # if problem not self adjoint, solve for adjoint variables and
                # calculate derivatives, else use analytical solution
                if self_adj is None:
                    dobj[:,0] += rhs_adj 
                    break
                elif self_adj:
                    #dobj[:] += rhs_adj
                    adj[free,i] = rhs_adj[free,0]
                else:
                    if problems is None:
                        adj[free,i:i+1],_,_ = solve_lin(K,
                                               rhs=rhs_adj[free,i:i+1],
                                               rhs0=adj[free,i:i+1],
                                               solver=lin_solver,
                                               solver_kw=lin_solver_kw,
                                               factorization=fact,
                                               P=precond,
                                               preconditioner=preconditioner,
                                               preconditioner_kw=preconditioner_kw)
                    else:
                        adj_out = _solver0.adjoint(rhs=rhs_adj,
                                                   state=state,
                                                   parameters=parameters,
                                                   solver_kw=solver_kw[0],
                                                   logger=log)
                        adj[:, i:i+1] = adj_out[f"adj_{_solver0.fieldname}"]
                if problems is None:
                    # standard contribution of element stiffness/conductivity
                    dobj_offset = np.matvec(KE,u[edofMat,i])
                    # contribution due to force induced by strain
                    if "strain_uniform" in body_forces_kw.keys():
                        dobj_offset -= fe_strain[None,:,i]
                    # generic density dependent element wise force
                    if f0 is not None:
                        dobj_offset -= f0[None,:,i]
                    #
                    dobj[:,0] += (matinterpol_dx(xPhys=xPhys, **matinterpol_kw)*\
                                 adj[edofMat,i]*dobj_offset).sum(axis=1)
                    # update sensitivity for quantities that do not need a small
                    # offset to avoid degeneracy of the FE problem
                    if "density_coupled" in body_forces_kw.keys():
                        dobj[:,0] -= simp_dx(xPhys=xPhys, eps=0., penal=1.)[:,0]*\
                                             np.dot(adj[edofMat,i],fe_dens[:,i])
                #
                log.debug("adj: it.: {0}, problem: {1}, min. adj: {2:.10f}, med. adj: {3:.10f}, max. adj: {4:.10f}".format(
                           loop,i,np.min(adj[:,i]),np.median(adj[:,i]),np.max(adj[:,i])))
            else:
                if problems is not None:
                    dsens = _solver0.sensitivity(state=state,
                                                 parameters=parameters,
                                                 adjoint=adj,
                                                 solver_kw=solver_kw[0],
                                                 logger=log)
                    dobj[:, 0] += dsens["dL_dxPhys"][:, 0]
        # optimizer is unknown.
        else:
            raise NotImplementedError("Unknown optimizer.")
        # constraints, constraint gradients and adjoint analysis
        constrs[:,:] = 0.
        dconstrs[:,:] = 0.
        for k, c in enumerate(constraints):
            val = 0.
            dconstr = np.zeros(x.shape, order="F")
            for j in np.arange(f.shape[1]):
                val, rhs_adj_c, self_adj_c = c["func"](obj=val,
                                                       i=j,
                                                       xPhys=xPhys,
                                                       u=u,
                                                       KE=KE,
                                                       edofMat=edofMat,
                                                       Kes=Kes,
                                                       matinterpol=matinterpol,
                                                       matinterpol_kw=matinterpol_kw,
                                                       mapping=mapping, 
                                                       invmapping=invmapping,
                                                       cellVolume=mesh["cellVolume"],
                                                       **c["kw"])
                # depends only on xPhys or any design variable: rhs_adj_c is already the gradient
                if self_adj_c is None:
                    dconstr[:,0] += rhs_adj_c.ravel()
                    break
                # self adjoint 
                elif self_adj_c:
                    adj[free,j] = rhs_adj_c[free,0]
                # adjoint problem needs to be solved
                else:
                    if problems is None:
                        adj[free,j:j+1],_,_ = solve_lin(K,
                                               rhs=rhs_adj_c[free,j:j+1],
                                               rhs0=adj[free,j:j+1],
                                               solver=lin_solver,
                                               solver_kw=lin_solver_kw,
                                               factorization=fact,
                                               P=precond,
                                               preconditioner=preconditioner,
                                               preconditioner_kw=preconditioner_kw)
                    else:
                        adj_out = _solver0.adjoint(rhs=rhs_adj_c,
                                                   state=state,
                                                   parameters=parameters,
                                                   solver_kw=solver_kw[0],
                                                   logger=log)
                        adj[:, j:j+1] = adj_out[f"adj_{_solver0.fieldname}"]
                if problems is None:
                    # accumulate sensitivity via chain rule through material interpolation
                    dconstr_offset = np.matvec(KE, u[edofMat,j])
                    if "strain_uniform" in body_forces_kw.keys():
                        dconstr_offset -= fe_strain[None,:,j]
                    if f0 is not None:
                        dconstr_offset -= f0[None,:,j]
                    dconstr[:,0] += (matinterpol_dx(xPhys=xPhys, **matinterpol_kw)*\
                                     adj[edofMat,j]*dconstr_offset).sum(axis=1)
                    if "density_coupled" in body_forces_kw.keys():
                        dconstr[:,0] -= simp_dx(xPhys=xPhys, eps=0., penal=1.)[:,0]*\
                                        np.dot(adj[edofMat,j],fe_dens[:,j])
            else:
                if problems is not None:
                    dsens_c = _solver0.sensitivity(state=state,
                                                   parameters=parameters,
                                                   adjoint=adj,
                                                   solver_kw=solver_kw[0],
                                                   logger=log)
                    dconstr[:, 0] += dsens_c["dL_dxPhys"][:, 0]
            # apply constraint type sign
            if c["type"] == "leq":
                constrs[k, 0] = val - c["value"]
                dconstrs[:, k:k+1] = dconstr
            elif c["type"] == "geq":
                constrs[k, 0] = c["value"] - val
                dconstrs[:, k:k+1] = -dconstr
            elif c["type"] == "eq":
                constrs[k, 0] = val - c["value"]
                dconstrs[:, k:k+1] = dconstr
        # optionally normalize each constraint row by a reference value
        for k, constraint in enumerate(constraints):
            if constraint["normalize"]:
                if constraint["norm_ref"] is None: 
                    ref = np.maximum(abs(c["value"]), c["norm_delta"])
                else:
                    ref = constraint["norm_ref"]
                constrs[k, 0] = constrs[k, 0]/ref
                dconstrs[:, k:k+1] = dconstrs[:, k:k+1]/ref
        #
        log.debug("Pre-Sensitivity Filter: it.: {0}, min(dobj): {1:.10f}, max(dobj): {2:.10f}, dv: {3:.10f}".format(
                  loop, 
                  np.min(dobj), 
                  np.max(dobj), 
                  np.min(dconstrs)))
        # sensitivity propagation (call it this way to avoid confusion with sensitvity filter)
        dobj, dconstrs = sensitivity_propagation(dobj=dobj,
                                                 dconstrs=dconstrs,
                                                 x=x,
                                                 xTilde=xTilde,
                                                 xPhys=xPhys,
                                                 ft=ft,
                                                 filter_kw=filter_kw,
                                                 el_flags_policy=el_flags_policy,
                                                 prescribed_mask=prescribed_mask)
        #
        log.debug("Post-Sensitivity Filter: it.: {0}, min(dobj): {1:.10f}, max(dobj): {2:.10f}, dv: {3:.10f}".format(
                  loop, 
                  np.min(dobj), np.max(dobj), 
                  np.min(dconstrs)))
        # scale objective and its gradient if requested
        if "scale_factor" in obj_kw and obj_kw["scale_factor"] is not None:
            obj   *= obj_kw["scale_factor"]
            dobj  *= obj_kw["scale_factor"]
        # scale each constraint and its gradient by its own factor if requested
        for k, c in enumerate(constraints):
            if c["scale_factor"] is not None:
                constrs[k, 0] *= c["scale_factor"]
                dconstrs[:, k:k+1] *= c["scale_factor"]
        # design variables update by optimizer
        if continuation_kw is not None:
            run_continuation(continuation_kw, stage=0,
                             filter_kw=filter_kw, optimizer_kw=optimizer_kw,
                             logger=log)
        # optimality criteria
        if optimizer=="oc":
            (x[:,0], g) = oc_top88(x=x[:,0], 
                                   volfrac=constraints[0]["value"],
                                   dc=dobj[:,0], 
                                   dv=dconstrs[:,0]*x[:,0].shape[0], 
                                   g=g,
                                   el_flags=el_flags)
        elif optimizer=="ocm":
            (x[:,0], g) = oc_mechanism(x=x[:,0], 
                                       volfrac=constraints[0]["value"],
                                       dc=dobj[:,0], 
                                       dv=dconstrs[:,0]*x[:,0].shape[0], 
                                       g=g,
                                       el_flags=el_flags)
        elif optimizer=="ocg":
            (x[:,0], g) = oc_generalized(x=x[:,0], 
                                         volfrac=constraints[0]["value"],
                                         dc=dobj[:,0], 
                                         dv=dconstrs[:,0]*x[:,0].shape[0], 
                                         g=g,
                                         el_flags=el_flags)
        # method of moving asymptotes
        elif optimizer=="mma":
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                                     n=x.shape[0],
                                                                     iter=loop,
                                                                     xval=x,
                                                                     xold1=hist["xhist"][-1],
                                                                     xold2=hist["xhist"][-2],
                                                                     f0val=obj,
                                                                     df0dx=dobj,
                                                                     fval=constrs,
                                                                     dfdx=dconstrs.T,
                                                                     **optimizer_kw)

            # update asymptotes
            optimizer_kw["low"] = low
            optimizer_kw["upp"] = upp
            x = xmma.copy()
        # globally convergent method of moving asymptotes
        elif optimizer == "gcmma":
            # update asymptotes and raa parameters
            optimizer_kw["low"], optimizer_kw["upp"], \
            optimizer_kw["raa0"], optimizer_kw["raa"] = asymp(
                outeriter=loop,
                n=x.shape[0],
                xval=x,
                xold1=hist["xhist"][-1],
                xold2=hist["xhist"][-2],
                df0dx=dobj,
                dfdx=dconstrs.T,
                **optimizer_kw)
            # first subproblem solve
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,f0app,fapp = gcmmasub(
                m=optimizer_kw["nconstr"],
                n=x.shape[0],
                iter=loop,
                xval=x,
                xold1=hist["xhist"][-1],
                xold2=hist["xhist"][-2],
                f0val=obj,
                df0dx=dobj,
                fval=constrs,
                dfdx=dconstrs.T,
                **optimizer_kw)
            # inner loop: tighten approximation until conservative
            u_inner = u.copy()
            for _inner in range(ninneriter):
                # apply forward filter to xmma without touching xTilde/xPhys
                xTilde_tmp = [t.copy() for t in xTilde]
                xTilde_tmp, xPhys_new = filter_design_variables(x=xmma,
                                                                xTilde=xTilde_tmp,
                                                                xPhys=xPhys.copy(),
                                                                ft=ft,
                                                                filter_kw=filter_kw,
                                                                el_flags_policy=el_flags_policy,
                                                                passive_mask=passive_mask,
                                                                active_mask=active_mask)
                # solve FEM at xmma
                u_inner, K_inner, fact_inner, precond_inner, Kes_new, _, _ = \
                    solve_phys_problems(xPhys=xPhys_new,
                                        KE=KE,
                                        iK=iK, jK=jK,
                                        ndof=ndof, n=n_el,
                                        edofMat=edofMat,
                                        f=f,
                                        fixed=fixed, free=free, springs=springs,
                                        matinterpol=matinterpol,
                                        matinterpol_kw=matinterpol_kw,
                                        assembly_mode=assembly_mode,
                                        assm_indcs=assm_indcs,
                                        body_forces_kw=body_forces_kw,
                                        fe_strain=fe_strain,
                                        fe_dens=fe_dens,
                                        lin_solver=lin_solver,
                                        lin_solver_kw=lin_solver_kw,
                                        preconditioner=preconditioner,
                                        preconditioner_kw=preconditioner_kw,
                                        u=u_inner)
                # evaluate objective at xmma
                obj_new = 0.
                for i in np.arange(f.shape[1]):
                    obj_new, _, self_adj_new = obj_func(obj=obj_new, 
                                                        i=i,
                                                        xPhys=xPhys_new, 
                                                        u=u_inner,
                                                        KE=KE, 
                                                        edofMat=edofMat, 
                                                        Kes=Kes_new,
                                                        matinterpol=matinterpol,
                                                        matinterpol_kw=matinterpol_kw,
                                                        mapping=mapping, 
                                                        invmapping=invmapping,
                                                        cellVolume=mesh["cellVolume"],
                                                        **obj_kw)
                    if self_adj_new is None:
                        break
                # evaluate constraint values at xmma
                constrs_new = np.zeros_like(constrs)
                for k, c in enumerate(constraints):
                    val_new = 0.
                    for j in np.arange(f.shape[1]):
                        val_new, _, self_adj_c_new = c["func"](obj=val_new, 
                                                               i=j,
                                                               xPhys=xPhys_new, u=u_inner,
                                                               KE=KE, 
                                                               edofMat=edofMat, 
                                                               Kes=Kes_new,
                                                               matinterpol=matinterpol,
                                                               matinterpol_kw=matinterpol_kw,
                                                               mapping=mapping, 
                                                               invmapping=invmapping,
                                                               cellVolume=mesh["cellVolume"],
                                                               **c["kw"])
                        if self_adj_c_new is None:
                            break
                    if c["type"] == "leq":
                        constrs_new[k, 0] = val_new - c["value"]
                    elif c["type"] == "geq":
                        constrs_new[k, 0] = c["value"] - val_new
                    elif c["type"] == "eq":
                        constrs_new[k, 0] = val_new - c["value"]
                # apply same normalization and scaling as in the main sensitivity block
                for k, c in enumerate(constraints):
                    if c["normalize"]:
                        ref = c["norm_ref"] if c["norm_ref"] is not None \
                              else np.maximum(abs(c["value"]), c["norm_delta"])
                        constrs_new[k, 0] /= ref
                if "scale_factor" in obj_kw and obj_kw["scale_factor"] is not None:
                    obj_new *= obj_kw["scale_factor"]
                for k, c in enumerate(constraints):
                    if c["scale_factor"] is not None:
                        constrs_new[k, 0] *= c["scale_factor"]
                # conservative check
                conserv = concheck(m=optimizer_kw["nconstr"],
                                   f0app=f0app,
                                   f0valnew=obj_new,
                                   fapp=fapp,
                                   fvalnew=constrs_new,
                                   **optimizer_kw)
                if conserv:
                    break
                # tighten approximation and resolve subproblem
                optimizer_kw["raa0"], optimizer_kw["raa"] = raaupdate(
                                                        xmma=xmma,
                                                        xval=x,
                                                        f0valnew=float(obj_new),
                                                        fvalnew=constrs_new,
                                                        f0app=f0app,
                                                        fapp=fapp,
                                                        **optimizer_kw)
                xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,f0app,fapp = gcmmasub(
                                                        m=optimizer_kw["nconstr"],
                                                        n=x.shape[0],
                                                        iter=loop,
                                                        xval=x,
                                                        xold1=hist["xhist"][-1],
                                                        xold2=hist["xhist"][-2],
                                                        f0val=obj,
                                                        df0dx=dobj,
                                                        fval=constrs,
                                                        dfdx=dconstrs.T,
                                                        **optimizer_kw)
                x = xmma.copy()
        #
        log.debug("Post Density Update: it.: {0}, med(x): {1:.10f}, mean(x): {2:.10f}, med(xPhys): {3:.10f}".format(
                   loop, np.median(x),np.mean(x), np.median(xPhys)))
        # mixing
        if ((loop-accelerator_kw["accel_start"])%accelerator_kw["accel_freq"])==0 \
            and loop >= accelerator_kw["accel_start"] and \
            accelerator_kw["accelerator"] is not None:
            x[:] = accelerator_kw["accelerator"](x=x.reshape( np.prod(x.shape), order="F" ),
                                                 xhist=[_x.reshape( np.prod(x.shape), order="F" ) for _x in hist["xhist"]],
                                                 **accelerator_kw).reshape(x.shape,order="F")
        elif mix is not None:
            x[:] = hist["xhist"][-1]*(1-mix) + x*mix
        #
        log.debug("Post Mixing Update: it.: {0}, med. x.: {1:.10f}, med. xPhys: {2:.10f}".format(
                  loop, np.median(x),np.median(xPhys)))
        # Filter design variables
        xTilde, xPhys = filter_design_variables(x=x,
                                                xTilde=xTilde,
                                                xPhys=xPhys,
                                                ft=ft,
                                                filter_kw=filter_kw,
                                                el_flags_policy=el_flags_policy,
                                                passive_mask=passive_mask,
                                                active_mask=active_mask)
        # update and prune history (after forward filter so xPhys_hist is consistent)
        update_history(**hist,
                       x=x, xPhys=xPhys, obj=obj, constrs=constrs,
                       max_history=max_history)
        #
        log.debug("Post Density Filter: it.: {0}, med. x.: {1:.10f}, med. xPhys: {2:.10f}".format(
                  loop, np.median(x),np.median(xPhys)))
        # compute the change
        change = convergence_kw["change_func"](**hist,**convergence_kw)
        # plot to screen
        if output_kw["display"]:
            if ndim == 2:
                plotfunc(mapping(-xPhys))
            fig.canvas.draw()
            plt.pause(0.01)
        #
        if output_kw["output_movie"]:
            export_vtk(filename="_".join([output_kw["file"],
                                   str(loop).zfill(output_kw["mov_ndigits"])]),
                       nelx=nelx,nely=nely,nelz=nelz,
                       xPhys=xPhys,x=x,
                       u=u,f=f,volfrac=volfrac)
        # write iteration history to screen (req. Python 2.6 or newer)
        log.info("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch.: {3:.10f} constrs.: [{4}]".format(
                     loop+1, obj, xPhys.mean(), change,
                     ", ".join(f"{c['name']}: {v:.6f}" for c, v in zip(constraints, constrs[:,0]))))
        # convergence and parameter continuation check
        if continuation_kw is None:
            if change < convergence_kw["conv_tol"]:
                break
        else:
            if run_continuation(continuation_kw, stage=1,
                                **hist,
                                change=change,
                                loop=loop,
                                filter_kw=filter_kw,
                                optimizer_kw=optimizer_kw,
                                conv_tol=convergence_kw["conv_tol"],
                                logger=log):
                break
    ### Optimization loop finished
    if output_kw["export"]:
        #
        nodal_variables = {"u": u, 
                           "f": f}
        if springs is not None:
            spring_array = np.zeros((u.shape[0],1))
            spring_array[springs[0],0] = springs[1]
            nodal_variables["springs"] = spring_array
        if "l" in obj_kw.keys() and obj_kw["l"].shape[0] == u.shape[0]:
            nodal_variables["l_obj"] = obj_kw["l"]
        #
        element_variables={"el_flags": el_flags, 
                           "xPhys": xPhys, 
                           "x": x}
        #
        export_vtk(filename=output_kw["file"],
                   nelx=nelx,nely=nely,nelz=nelz,
                   volfrac=volfrac, 
                   elem_size=l,
                   nodal_variables=nodal_variables,
                   element_variables=element_variables)
    # finish profiling
    if output_kw["profile"]:
        profiler.disable()
        profiler.dump_stats(output_kw["file"]+".prof")
    #
    if output_kw["display"]:
        plt.show()
        input("Press any key...")
    #
    if output_kw["display"] and output_kw["save_pdf"]:
        fig.savefig(output_kw["file"]+".pdf", 
                    **output_kw["pdf_kw"])
    return x, xTilde, xPhys, obj