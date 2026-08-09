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
# import workflow routines
from topoptlab.topology_opt.mesh import create_mesh 
from topoptlab.topology_opt.design import initialize_design
from topoptlab.topology_opt.history import initialize_history, update_history
from topoptlab.topology_opt.optimizer import initialize_optimizer
from topoptlab.topology_opt.solver_routines import adjoint_loop, initialize_problems, solver_loop
from topoptlab.topology_opt.plotting import initialize_plotting

def initialize_materialinterpolation(
        matinterpol_kw: Union[None, Dict, List[Union[Dict, List[Dict]]]],
        problems: List[Union[ProblemSolver, List[ProblemSolver]]]
        ) -> List[Union[Dict, List[Dict]]]:
    """
    Build per-problem interpolation keyword dicts from solver defaults and
    user overrides.

    Parameters
    ----------
    matinterpol_kw : None, dict, or list
        Per-property call-time kwargs (e.g. ``{"Young's modulus": {"eps": 1e-9,
        "penal": 3.}}``).  Accepted forms:

        ``None``
            Use each solver's stored defaults with no overrides.
        ``dict``
            One shared dict applied to every problem group.
        ``list``
            One entry per problem group: a ``dict`` for single or weakly/
            strongly coupled solvers, or a ``list`` of dicts (one per solver)
            for monolithic groups.

    problems : list
        Initialized problem groups.  Each entry is either a single
        ``ProblemSolver`` or a list of ``ProblemSolver`` instances
        (monolithic coupling).

    Returns
    -------
    list
        One entry per problem group: a ``dict`` for single/coupled solvers or
        a ``list`` of dicts for monolithic groups.
    """
    if isinstance(matinterpol_kw, list):
        if len(problems) != len(matinterpol_kw):
            raise ValueError("len(problems) != len(matinterpol_kw): ",
                             len(problems), len(matinterpol_kw))
        for i, (problem, mat_kw) in enumerate(zip(problems, matinterpol_kw)):
            if isinstance(problem, ProblemSolver):
                if not isinstance(mat_kw, dict):
                    raise TypeError(f"problem {i}'s matinterpol_kw is not a dict: ",
                                    type(mat_kw))
            elif isinstance(problem, list):
                if not isinstance(mat_kw, list):
                    raise TypeError(f"problem {i}'s matinterpol_kw is not a list: ",
                                    type(mat_kw))
                if len(problem) != len(mat_kw):
                    raise ValueError(f"len(problems[{i}]) != len(matinterpol_kw[{i}]): ",
                                     len(problem), len(mat_kw))
                for j, (prob, mkw) in enumerate(zip(problem, mat_kw)):
                    if not isinstance(prob, ProblemSolver):
                        raise TypeError(f"problems[{i}][{j}] is not a ProblemSolver: ",
                                        type(prob))
                    if not isinstance(mkw, dict):
                        raise TypeError(f"matinterpol_kw[{i}][{j}] is not a dict: ",
                                        type(mkw))
        # build from solver defaults + user overrides
        interpol_kw = []
        for problem, mat_kw in zip(problems, matinterpol_kw):
            if isinstance(problem, list):
                group_kw = []
                for s, mkw in zip(problem, mat_kw):
                    kw = dict(s.default_interpol_kw)
                    kw.update(mkw)
                    group_kw.append(kw)
                interpol_kw.append(group_kw)
            else:
                kw = dict(problem.default_interpol_kw)
                kw.update(mat_kw)
                interpol_kw.append(kw)
    elif isinstance(matinterpol_kw, dict):
        interpol_kw = []
        for problem in problems:
            if isinstance(problem, list):
                group_kw = []
                for s in problem:
                    kw = dict(s.default_interpol_kw)
                    kw.update(matinterpol_kw)
                    group_kw.append(kw)
                interpol_kw.append(group_kw)
            else:
                kw = dict(problem.default_interpol_kw)
                kw.update(matinterpol_kw)
                interpol_kw.append(kw)
    elif matinterpol_kw is None:
        interpol_kw = []
        for problem in problems:
            if isinstance(problem, list):
                interpol_kw.append([dict(s.default_interpol_kw) for s in problem])
            else:
                interpol_kw.append(dict(problem.default_interpol_kw))
    else:
        raise TypeError("matinterpol_kw must be None, a dict, or a list; "
                        f"got {type(matinterpol_kw)}")
    return interpol_kw

# MAIN DRIVER
def main(nelx: int, nely: int,
         volfrac: Union[float, np.ndarray], #penal: float,
         rmin: float, 
         ft: [int,TOFilter,List[TOFilter]] = 1,
         filter_kw: Union[Dict,List] = {},
         simulation_kw: Dict = {"type": "stationary",
                                "coordinate_system": "cartesian"},
         nelz: Union[None,int] = None,
         initial_guess: Union[None,Dict[str, np.ndarray]] = None,
         design_parameterization_kw: Dict = {},
         filter_mode: str = "convolution",
         lin_solver_kw: Union[Dict, List[Dict]] = {"name": "scipy-direct"},
         preconditioner_kw: Dict = {"name": None},
         assembly_mode: str = "full",
         materials_kw: Union[Dict, List[Dict]] = [{"Young's modulus": 1.,
                                                   "Poisson's ratio": 0.3}],
         bcs: Callable = mbb_2d,
         problems: Union[None, List[Union[Callable, List[Callable]]]] = None,
         solver_kw: Union[Dict, List[Dict]] = {},
         solver_coupling: Union[None, List[str]] = None,
         lk: Union[None,Callable] = None,
         l: Union[float,List,np.ndarray] = 1.,
         obj_func: Callable = compliance, 
         obj_kw: Dict = {},
         matinterpol: Callable = simp,
         matinterpol_dx: Callable = simp_dx,
         matinterpol_kw: Union[None, Dict, List[Dict]] = None,
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
    materials_kw : list of dict
        One dict per material, each containing the material constants (e.g.
        ``"Young's modulus"``, ``"Poisson's ratio"``).  A single dict is
        accepted for backward compatibility and treated as a one-material list.
        ``n_mat`` is inferred as ``len(materials_kw)`` after normalization.
        Must be consistent with ``log_material_properties`` in log_utils.py.
    lin_solver_kw : dict or list of dict
        Linear solver configuration.  Must contain ``"name"`` (solver
        backend).  When ``problems`` is used, a list of dicts (one per
        problem) is accepted so that the linear solver can be changed per
        problem or updated by a continuation strategy between iterations.
    solver_kw: dict or list of dict
        all information for the specific ProblemSolver at hand. Usually gives details
        about physical discretization, which effects to include etc.
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
    if isinstance(lin_solver_kw, dict):
        lin_solver_kw = len(problems)*[lin_solver_kw]
    if isinstance(preconditioner_kw, dict): 
        preconditioner_kw = len(problems)*[preconditioner_kw]
    if isinstance(solver_kw, dict): 
        solver_kw = len(problems)*[solver_kw]
    # normalize materials_kw: single dict → one-element list
    if isinstance(materials_kw, dict):
        materials_kw = [materials_kw]
    #
    n_mat = len(materials_kw)
    simulation_kw["n_mat"] = n_mat
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
    # bundle all mesh-related information; irregular-mesh support will extend this
    mesh_kw = create_mesh(nelx = nelx,  
                          nely = nely, 
                          nelz = nelz, 
                          l = l,
                          mesh_file = None, 
                          logger = log)
    mapping, invmapping = mesh_kw["mapping"], mesh_kw["invmapping"]
    l = mesh_kw["l"]
    # Allocate design variables (as array), initialize and allocate sens.
    x, xPhys = initialize_design(n=n_el,
                                 initial_guess=initial_guess,
                                 volfrac=volfrac,
                                 n_mat=n_mat,
                                 **design_parameterization_kw)
    # initialize problem solvers now that logger and mesh dims are known
    # canonicalize bcs 
    problems = initialize_problems(problems=problems, 
                                   bcs=bcs, 
                                   mesh_kw=mesh_kw, 
                                   solver_kw=solver_kw,
                                   logger=log)
    # precompute prescribed-element masks for filter policy corrections
    passive_mask, active_mask, prescribed_mask = None, None, None
    if el_flags is not None:
        passive_mask = el_flags == 1
        active_mask = el_flags == 2
        prescribed_mask = (el_flags == 1) | (el_flags == 2) 
    # initialize arrays for gradients
    dobj = np.zeros(xPhys.shape, order="F")
    # build and validate constraint list
    if constraints is None:
        constraints = []
    # legacy: if volfrac is given, prepend one volume fraction constraint per material
    if volfrac is not None:
        volfrac_arr = np.atleast_1d(volfrac)
        vf_constraints = [{"name": f"volume fraction {i}",
                           "func": volume_fraction_constraint,
                           "type": "leq",
                           "value": float(volfrac_arr[i] if len(volfrac_arr) > 1 else volfrac_arr[0]),
                           "kw":   {"col_ind": i}}
                          for i in range(n_mat)]
        if ft == 0:
            for c in vf_constraints:
                c["filter"] = False
        constraints = vf_constraints + list(constraints)
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
    dconstrs = np.zeros((n_el * xPhys.shape[1], n_constr), order="F")
    # initialize optimizer state
    optimizer_kw, x, xPhys, max_history = initialize_optimizer(
                                             optimizer=optimizer,
                                             optimizer_kw=optimizer_kw,
                                             x=x,
                                             xPhys=xPhys,
                                             n_constr=n_constr,
                                             n_el=n_el,
                                             ft=ft,
                                             el_flags=el_flags,
                                             prescribed_mask=prescribed_mask,
                                             volfrac=volfrac)
    # prepare FEM data structures
    ### this later needs to get deleted
    # mesh/DOF info pulled from the first solver — all setup was done at construction
    if isinstance(problems[0], ProblemSolver):
        _solver0 = problems[0]  
    else: 
        _solver0 = problems[0][0]
    f = _solver0.f        # load vector shape drives adj and load-case loop
    free = _solver0.free     # needed for self-adjoint branch
    KE = _solver0._KE0    # forwarded to obj_func
    edofMat = _solver0.edofMat  # forwarded to obj_func
    u = np.zeros_like(f) # placeholder; overwritten from state each iteration
    #
    if solver_coupling is None:
        solver_coupling = ["weak"] * len(problems)
    if isinstance(solver_kw, dict):
        solver_kw = [solver_kw] * len(problems)
    if isinstance(lin_solver_kw, dict):
        lin_solver_kw = [lin_solver_kw] * len(problems)
    #
    state = {}
    #
    interpol_kw = []
    for i, prob in enumerate(problems):
        if isinstance(prob, list):
            s = prob[0]
        else:
            s = prob
        kw = dict(s.default_interpol_kw)
        if matinterpol_kw is None:
            pass
        elif isinstance(matinterpol_kw, list):
            kw.update(matinterpol_kw[i])
        else:
            kw.update(matinterpol_kw)
        interpol_kw.append(kw)
    #
    parameters = {"xPhys": xPhys}
    # initialize filters
    ft = prepare_filters(ft=ft,
                         **mesh_kw,
                         filter_mode=filter_mode,
                         rmin=rmin,
                         n_constr=n_constr,
                         el_flags=el_flags,
                         el_flags_policy=el_flags_policy,
                         filter_kw=filter_kw,
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
    # initialize display functions
    if output_kw["display"]:
        fig, plotfunc = initialize_plotting(xPhys=xPhys, 
                                            **mesh_kw)
    #
    if output_kw["output_movie"]:
        output_kw["mov_ndigits"] = len(str(nouteriter))
    # initialize iteration history
    max_history, continuation_kw, hist = initialize_history(
                                            max_history=max_history,
                                            accelerator_kw=accelerator_kw,
                                            continuation_kw=continuation_kw,
                                            convergence_kw=convergence_kw,
                                            x=x,
                                            xPhys=xPhys,
                                            constrs=constrs)
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
        ### solve physical problems
        parameters["xPhys"] = xPhys
        state = solver_loop(problems,
                            state,
                            ntimesteps=1,
                            coupling=solver_coupling,
                            parameters=parameters,
                            solver_kw=solver_kw,
                            lin_solver_kw=lin_solver_kw,
                            preconditioner_kw=preconditioner_kw,
                            interpol_kw=interpol_kw,
                            logger=log)
        u = state[_solver0.fieldname]
        Kes = _solver0._terms["Kes"]
        #
        for i in range(f.shape[1]): 
            log.debug("FEM: it.: {0}, problem: {1}, min. u: {2:.12f}, med. u: {3:.12f}, max. u: {4:.12f}".format(
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
                                            cellVolume=mesh_kw["cellVolume"],
                                            **obj_kw)
            # if problem not self adjoint, solve for adjoint variables and
            # calculate derivatives, else use analytical solution
            if self_adj is None:
                dobj += rhs_adj.reshape(dobj.shape, order='F')
                break
            elif self_adj:
                #dobj[:] += rhs_adj
                adj[free,i] = rhs_adj[free,0]
            else:
                adj_out = adjoint_loop(problems,
                                        state,
                                        adj_rhs={_solver0.fieldname: rhs_adj},
                                        ntimesteps=1,
                                        coupling=solver_coupling,
                                        parameters=parameters,
                                        solver_kw=solver_kw,
                                        lin_solver_kw=lin_solver_kw,
                                        preconditioner_kw=preconditioner_kw,
                                        interpol_kw=interpol_kw,
                                        logger=log)
                adj[:, i:i+1] = adj_out[f"adj_{_solver0.fieldname}"]
            #
            log.debug("adj: it.: {0}, problem: {1}, min. adj: {2:.12f}, med. adj: {3:.12f}, max. adj: {4:.12f}".format(
                        loop,i,np.min(adj[:,i]),np.median(adj[:,i]),np.max(adj[:,i])))

            dobj[:] += _solver0.sensitivity(state=state,
                                            parameters=parameters,
                                            adjoint=adj,
                                            solver_kw=solver_kw[0],
                                            logger=log)["dL_dxPhys"]
        # constraints, constraint gradients and adjoint analysis
        constrs[:] = 0.
        dconstrs[:] = 0.
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
                                                       cellVolume=mesh_kw["cellVolume"],
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
                    adj_out = adjoint_loop(problems,
                                           state,
                                           adj_rhs={_solver0.fieldname: rhs_adj_c},
                                           ntimesteps=1,
                                           coupling=solver_coupling,
                                           parameters=parameters,
                                           solver_kw=solver_kw,
                                           lin_solver_kw=lin_solver_kw,
                                           preconditioner_kw=preconditioner_kw,
                                           interpol_kw=interpol_kw,
                                           logger=log)
                    adj[:, j:j+1] = adj_out[f"adj_{_solver0.fieldname}"]

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
            obj *= obj_kw["scale_factor"]
            dobj *= obj_kw["scale_factor"]
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
            if n_mat > 1:
                raise NotImplementedError("OC optimizer only supports single-material (n_mat=1).")
            (x[:,0], optimizer_kw["g"]) = oc_top88(x=x[:,0],
                                                    volfrac=constraints[0]["value"],
                                                    dc=dobj[:,0],
                                                    dv=dconstrs[:,0]*x[:,0].shape[0],
                                                    g=optimizer_kw["g"],
                                                    el_flags=el_flags)
        elif optimizer=="ocm":
            if n_mat > 1:
                raise NotImplementedError("OC optimizer only supports single-material (n_mat=1).")
            (x[:,0], optimizer_kw["g"]) = oc_mechanism(x=x[:,0],
                                                        volfrac=constraints[0]["value"],
                                                        dc=dobj[:,0],
                                                        dv=dconstrs[:,0]*x[:,0].shape[0],
                                                        g=optimizer_kw["g"],
                                                        el_flags=el_flags)
        elif optimizer=="ocg":
            if n_mat > 1:
                raise NotImplementedError("OC optimizer only supports single-material (n_mat=1).")
            (x[:,0], optimizer_kw["g"]) = oc_generalized(x=x[:,0],
                                                          volfrac=constraints[0]["value"],
                                                          dc=dobj[:,0],
                                                          dv=dconstrs[:,0]*x[:,0].shape[0],
                                                          g=optimizer_kw["g"],
                                                          el_flags=el_flags)
        # method of moving asymptotes
        elif optimizer=="mma":
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                                     n=x.size,
                                                                     iter=loop,
                                                                     xval=x.reshape(-1, 1, order='F'),
                                                                     xold1=hist["xhist"][-1].reshape(-1, 1, order='F'),
                                                                     xold2=hist["xhist"][-2].reshape(-1, 1, order='F'),
                                                                     f0val=obj,
                                                                     df0dx=dobj.reshape(-1, 1, order='F'),
                                                                     fval=constrs,
                                                                     dfdx=dconstrs.reshape(-1, n_constr, order='F').T,
                                                                     **optimizer_kw)

            # update asymptotes
            optimizer_kw["low"] = low
            optimizer_kw["upp"] = upp
            x = xmma.reshape(n_el, -1, order='F')
        # globally convergent method of moving asymptotes
        elif optimizer == "gcmma":
            # update asymptotes and raa parameters
            optimizer_kw["low"], optimizer_kw["upp"], \
            optimizer_kw["raa0"], optimizer_kw["raa"] = asymp(
                outeriter=loop,
                n=x.size,
                xval=x.reshape(-1, 1, order='F'),
                xold1=hist["xhist"][-1].reshape(-1, 1, order='F'),
                xold2=hist["xhist"][-2].reshape(-1, 1, order='F'),
                df0dx=dobj.reshape(-1, 1, order='F'),
                dfdx=dconstrs.reshape(-1, n_constr, order='F').T,
                **optimizer_kw)
            # first subproblem solve
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,f0app,fapp = gcmmasub(
                m=optimizer_kw["nconstr"],
                n=x.size,
                iter=loop,
                xval=x.reshape(-1, 1, order='F'),
                xold1=hist["xhist"][-1].reshape(-1, 1, order='F'),
                xold2=hist["xhist"][-2].reshape(-1, 1, order='F'),
                f0val=obj,
                df0dx=dobj.reshape(-1, 1, order='F'),
                fval=constrs,
                dfdx=dconstrs.reshape(-1, n_constr, order='F').T,
                **optimizer_kw)
            # inner loop: tighten approximation until conservative
            state_inner = dict(state)
            for _inner in range(ninneriter):
                # apply forward filter to xmma without touching xTilde/xPhys
                if xTilde is not None:
                    xTilde_tmp = [t.copy() for t in xTilde]
                else:
                    xTilde_tmp = None
                xTilde_tmp, xPhys_new = filter_design_variables(x=xmma,
                                                                xTilde=xTilde_tmp,
                                                                xPhys=xPhys.copy(),
                                                                ft=ft,
                                                                filter_kw=filter_kw,
                                                                el_flags_policy=el_flags_policy,
                                                                passive_mask=passive_mask,
                                                                active_mask=active_mask)
                # solve FEM at xmma
                parameters_inner = {**parameters, "xPhys": xPhys_new}
                state_inner = solver_loop(problems,
                                          state_inner,
                                          ntimesteps=1,
                                          coupling=solver_coupling,
                                          parameters=parameters_inner,
                                          solver_kw=solver_kw,
                                          lin_solver_kw=lin_solver_kw,
                                          preconditioner_kw=preconditioner_kw,
                                          interpol_kw=interpol_kw,
                                          logger=log)
                u_inner = state_inner[_solver0.fieldname]
                Kes_new = _solver0._terms["Kes"]
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
                                                        cellVolume=mesh_kw["cellVolume"],
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
                                                               cellVolume=mesh_kw["cellVolume"],
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
                                                        xval=x.reshape(-1, 1, order='F'),
                                                        f0valnew=float(obj_new),
                                                        fvalnew=constrs_new,
                                                        f0app=f0app,
                                                        fapp=fapp,
                                                        **optimizer_kw)
                xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,f0app,fapp = gcmmasub(
                                                        m=optimizer_kw["nconstr"],
                                                        n=x.size,
                                                        iter=loop,
                                                        xval=x.reshape(-1, 1, order='F'),
                                                        xold1=hist["xhist"][-1].reshape(-1, 1, order='F'),
                                                        xold2=hist["xhist"][-2].reshape(-1, 1, order='F'),
                                                        f0val=obj,
                                                        df0dx=dobj.reshape(-1, 1, order='F'),
                                                        fval=constrs,
                                                        dfdx=dconstrs.reshape(-1, n_constr, order='F').T,
                                                        **optimizer_kw)
            x = xmma.reshape(n_el, -1, order='F')
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
            plotfunc(mapping(-xPhys))
            fig.canvas.draw()
            plt.pause(0.001)
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
    # treshold design
    #xThresh = threshold(xPhys,volfrac)
    # TO DO: compute and log performance on thresholded design

    # export design
    if output_kw["export"]:
        #
        nodal_variables = {"u": u, 
                           "f": f}
        #
        springs = None
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