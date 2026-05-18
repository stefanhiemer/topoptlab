# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Tuple, Union
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
from topoptlab.filter.fetch_filter import fetch_filters
from topoptlab.filter.convolution_filter import assemble_convolution_filter
from topoptlab.filter.helmholtz_filter import assemble_helmholtz_filter
from topoptlab.filter.matrix_filter import assemble_matrix_filter
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
from mmapy import mmasub

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
         filter_mode: str = "matrix",
         lin_solver_kw: Dict = {"name": "scipy-direct"}, 
         preconditioner_kw: Dict = {"name": None},
         assembly_mode: str = "full",
         materials_kw: Dict = {"E": 1.}, 
         body_forces_kw: Dict = {},
         bcs: Callable = mbb_2d, 
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
         nouteriter: int = 2000, ninneriter: int = 15,
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
        array of flags/integers that switch behaviour of specific elements.
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
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
    #
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
    else:
        ndim = 3
        create_edofMat = create_edofMat3d
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
    n = np.prod([nelx,nely,nelz][:ndim])
    #
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # only needed for homogenization so far
    cellVolume = np.prod(l)*n
    # get function of element stiffness matrix
    if lk is None and ndim == 2:
        lk = lk_linear_elast_2d
    elif lk is None and ndim == 3:
        lk = lk_linear_elast_3d
    # Allocate design variables (as array), initialize and allocate sens.
    if not isinstance(initial_guess,dict):
        initialguess_keys = None
    else:
        initialguess_keys = initial_guess.keys()
    #
    if initialguess_keys is None or "x" not in initialguess_keys:
        x = volfrac * np.ones( (n,1), dtype=float,order='F')
    else:
        x = initial_guess["x"]
    #
    if initialguess_keys is None or "xPhys" not in initialguess_keys:
        xPhys = x.copy()
    else:
        xPhys = initial_guess["xPhys"]
    # intermediate filter variables
    if isinstance(ft, list):
        xTilde = []
        for i in range(len(ft) - 1):
            key = f" xTilde-{i}"
            if initialguess_keys is None or key not in initialguess_keys:
                xTilde.append(x.copy())
            else:
                xTilde.append(initial_guess[key])
    else:
        xTilde = None
    # precompute prescribed-element masks for filter policy corrections
    passive_mask    = None
    active_mask     = None
    prescribed_mask = None
    if el_flags is not None:
        passive_mask    = el_flags == 1
        active_mask     = el_flags == 2
        prescribed_mask = el_flags != 0
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
    dconstrs = np.zeros( (n, n_constr) )
    # initialize history length needed for optimizer
    optimizer_kw = check_optimizer_kw(optimizer=optimizer,
                                      n=x.shape[0],
                                      ft=ft,
                                      n_constr=n_constr,
                                      optimizer_kw=optimizer_kw)
    if optimizer in ["oc","ocm"]:
        # legacy OC: only a single constraint (volume fraction) is supported
        if n_constr != 1:
            raise ValueError(
                f"Optimizers 'oc' and 'ocm' support exactly one constraint, "
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
            x[mask] = 1.
            xPhys[mask] = 1.
            # redistribute volfrac over free elements so that passive elements
            # (contributing 0) add material to the free set
            if volfrac is not None:
                n_free = (~prescribed_mask).sum()
                x_free = np.clip(volfrac * n / n_free, 0., 1.)
                x[~prescribed_mask, 0]     = x_free
                xPhys[~prescribed_mask, 0] = x_free
    else:
        raise ValueError("Unknown optimizer: ", optimizer)
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
    # function to convert densities, etc. to images/voxels for plotting or the
    # convolution filter.
    if ndim == 2:
        mapping = partial(map_eltoimg,
                          nelx=nelx,nely=nely)
    elif ndim == 3:
        mapping = partial(map_eltovoxel,
                          nelx=nelx,nely=nely,nelz=nelz)
    # prepare functions to invert this mapping if we use the convolution filter
    if not isinstance(ft, (int,list)) and issubclass(ft, TOFilter):
        ft = [ft(nelx=nelx,nely=nely,nelz=nelz,
                 filter_mode=filter_mode,
                 rmin=rmin,
                 n_constr=n_constr,
                 l=l,
                 el_flags=el_flags,
                 el_flags_policy=el_flags_policy,
                 **filter_kw)]
    elif isinstance(ft, list):
        ft = [ft_obj(nelx=nelx,nely=nely,nelz=nelz,
                     filter_mode=filter_mode,
                     rmin=rmin,
                     n_constr=n_constr,
                     l=l,
                     el_flags=el_flags,
                     el_flags_policy=el_flags_policy,
                     **filter_kw) \
              for ft_obj in ft]
    elif isinstance(ft, int) and ft == 0:
        # convert integer ft code to list of TOFilter objects
        # NOTE: temporary — all filters share the same kw; in general each
        # filter may receive its own keyword dict via filter_args
        _ft_kw = dict(nelx=nelx, nely=nely, nelz=nelz,
                      filter_mode=filter_mode,
                      rmin=rmin,
                      n_constr=n_constr,
                      l=l,
                      el_flags=el_flags,
                      el_flags_policy=el_flags_policy,
                      **filter_kw)
        ft = fetch_filters(ft=ft, filter_args=[_ft_kw, _ft_kw])
    elif isinstance(ft, int) and ft >= 1:
        # convert integer ft code to list of TOFilter objects
        # NOTE: temporary — all filters share the same kw; in general each
        # filter should receive its own keyword dict via filter_args
        _ft_kw = dict(nelx=nelx, nely=nely, nelz=nelz,
                      filter_mode=filter_mode,
                      rmin=rmin,
                      n_constr=n_constr,
                      l=l,
                      constraint_filter_mask=_constr_filter_mask,
                      el_flags=el_flags,
                      el_flags_policy=el_flags_policy,
                      **filter_kw)
        ft = fetch_filters(ft=ft, filter_args=[_ft_kw, _ft_kw])
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    elif filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                      rmin=rmin,ndim=ndim)
    elif filter_mode == "convolution":
        
        if ndim == 2:
            invmapping = partial(map_imgtoel,
                                 nelx=nelx,nely=nely)
        elif ndim == 3:
            invmapping = partial(map_voxeltoel,
                                 nelx=nelx,nely=nely,nelz=nelz)
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                           rmin=rmin,
                                           mapping=mapping,
                                           invmapping=invmapping)
    elif filter_mode == "helmholtz" and ft in [0,1]:
        KF,TF = assemble_helmholtz_filter(nelx=nelx,nely=nely,nelz=nelz,
                                          rmin=rmin, l=l,
                                          n1=n1,n2=n2,n3=n3,n4=n4)
        # LU decomposition. returns a function for solving, not the matrices
        lu_solve = factorized(KF)
    else:
        raise ValueError(f"Unknown filter. ft: {ft} filter_mode {filter_mode}")
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx,nely=nely,nelz=nelz,
                                 ndof=ndof)
    f0 = None
    # check that boundary conditions and body forces are compatible
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
    #
    # optimization loop
    for loop in np.arange(nouteriter):
        #
        # solve FEM, calculate obj. func. and gradients.
        if optimizer in ["oc","mma", "ocm","ocg"] or\
           (optimizer in ["gcmma"] and ninneriter==0) or\
           loop==0:
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
            ### Setup and solve FE problem
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
                                                cellVolume=cellVolume, 
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
                    adj[free,i:i+1],_,_ = solve_lin(K, 
                                           rhs=rhs_adj[free,i:i+1],
                                           rhs0=adj[free,i:i+1],
                                           solver=lin_solver,
                                           solver_kw=lin_solver_kw,
                                           factorization=fact,
                                           P=precond,
                                           preconditioner=preconditioner,
                                           preconditioner_kw=preconditioner_kw)
                # update sensitivity for quantities that need a small offset to
                # avoid degeneracy of the FE pro i need to add a continue after this I guess?blem
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
                log.debug("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                           loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        # optimizer is unknown.
        else:
            raise NotImplementedError("Unknown optimizer.")
        # Constraints, constraint gradients and adjoint analysis
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
                                                       cellVolume=cellVolume,
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
                    adj[free,j:j+1],_,_ = solve_lin(K,
                                               rhs=rhs_adj_c[free,j:j+1],
                                               rhs0=adj[free,j:j+1],
                                               solver=lin_solver,
                                               solver_kw=lin_solver_kw,
                                               factorization=fact,
                                               P=precond,
                                               preconditioner=preconditioner,
                                               preconditioner_kw=preconditioner_kw)
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
                  np.min(dobj), np.max(dobj), 
                  np.min(dconstrs)))
        # sensitivity filtering
        if isinstance(ft, list):
            #
            if ft[-1].filter_objective:
                dobj[:] = ft[-1].apply_filter_dx(x=x if len(ft)==1 else xTilde[-1],
                                                 x_filtered=xPhys,
                                                 dx_filtered=dobj, 
                                                 **filter_kw)
            #
            if np.any(ft[-1].constraint_filter_mask):
                dconstrs[:,ft[-1].constraint_filter_mask] = \
                     ft[-1].apply_filter_dx(x=x if len(ft)==1 else xTilde[-1], 
                                            x_filtered=xPhys,
                                            dx_filtered=dconstrs[:,ft[-1].constraint_filter_mask], 
                                            **filter_kw)
            if len(ft) > 1:
                for i in range(len(ft)-2,-1,-1):
                    #
                    if ft[i].filter_objective:
                        dobj[:] = ft[i].apply_filter_dx(x=x if i==0 else xTilde[i-1],
                                                        x_filtered=xTilde[i],
                                                        dx_filtered=dobj, 
                                                        **filter_kw)
                    #
                    if np.any(ft[i].constraint_filter_mask):
                        dconstrs[:,ft[i].constraint_filter_mask] =\
                            ft[i].apply_filter_dx(x=x if i==0 else xTilde[i-1], 
                                                  x_filtered=xTilde[i],
                                                  dx_filtered=dconstrs[:,ft[i].constraint_filter_mask], 
                                                  **filter_kw)
        elif ft == 0 and filter_mode == "matrix":
            dobj[:] = np.asarray(H@(x*dobj) /
                                 Hs) / np.maximum(0.001, x)
            #dobj[:] = H @ (dc*x) / Hs / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "convolution":
            dobj[:] = invmapping( convolve(mapping(x*dobj),
                                           weights=h, axes=(0,1,2)[:ndim],
                                           mode="constant",
                                           cval=0.0)) / hs / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "helmholtz":
            dobj[:] = TF.T @ lu_solve(TF@(dobj*xPhys))/np.maximum(0.001, x)
        elif ft == 1 and filter_mode == "matrix":
            dobj[:] = np.asarray(H*(dobj/Hs))
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        elif ft == 1 and filter_mode == "convolution":
            dobj[:] = invmapping( convolve(mapping(dobj),
                                           weights=h, axes=(0,1,2)[:ndim],
                                           mode="constant",
                                           cval=0.0)) / hs
            dconstrs[:] = invmapping( convolve(mapping(dconstrs),
                                           weights=h, axes=(0,1,2)[:ndim],
                                           mode="constant",
                                           cval=0.0)) / hs
        elif ft == 1 and filter_mode == "helmholtz":
            dobj[:] = TF.T @ lu_solve(TF@dobj)
            dconstrs[:] = TF.T @ lu_solve(TF@dconstrs)
        elif ft == -1:
            pass
        else:
            raise ValueError("No filter applied. ft: ", ft)
        # backward filter policy: zero sensitivities at prescribed elements
        if el_flags_policy is not None and el_flags_policy["correct_backward"]:
            dobj[prescribed_mask]     = 0.
            dconstrs[prescribed_mask] = 0.
        #
        log.debug("Post-Sensitivity Filter: it.: {0}, min(dobj): {1:.10f}, max(dobj): {2:.10f}, dv: {3:.10f}".format(
                  loop, 
                  np.min(dobj), np.max(dobj), 
                  np.min(dconstrs)))
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
                                       dv=dconstrs[:,0], 
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
        if isinstance(ft, list):
            if len(ft) > 1:
                xTilde[0] = ft[0].apply_filter(x=x,
                                               **filter_kw)
                for i in range(1,len(ft)-1):
                    xTilde[i] = ft[i].apply_filter(x=xTilde[i-1],
                                                   **filter_kw)
                xPhys = ft[-1].apply_filter(x=xTilde[-1],
                                            **filter_kw)
            else:
                xPhys = ft[0].apply_filter(x=x,
                                           **filter_kw)
            for _f in ft:
                if _f.changes_filter_kw:
                    _f.update_filter_kw(filter_kw)
        elif ft == 0:
            xPhys[:] = x
        elif ft == 1 and filter_mode == "matrix":
            xPhys[:] = np.asarray(H*x/Hs)
        elif ft == 1 and filter_mode == "convolution":
            xPhys[:] = invmapping( convolve(mapping(x),
                                           weights=h, axes=(0,1,2)[:ndim],
                                           mode="constant",
                                           cval=0.0)) / hs
        elif ft == 1 and filter_mode == "helmholtz":
            xPhys[:] = TF.T @ lu_solve(TF@x)
        elif ft == -1:
            xPhys[:]  = x
        # forward filter policy: restore prescribed densities after filtering
        if el_flags_policy is not None and el_flags_policy["correct_forward"]:
            xPhys[passive_mask] = 0.
            xPhys[active_mask]  = 1.
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
                     ", ".join(f"{v:.6f}" for v in constrs[:,0])))
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
    #
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