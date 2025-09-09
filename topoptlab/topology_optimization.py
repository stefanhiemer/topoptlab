# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Tuple, Union
from functools import partial
#
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.ndimage import convolve
#
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# functions to create filters
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
from topoptlab.optimizer.mma_utils import mma_defaultkws,gcmma_defaultkws
from topoptlab.objectives import compliance
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel,check_simulation_params
# logging related stuff
from topoptlab.log_utils import init_logging
#
from mmapy import mmasub

# MAIN DRIVER
def main(nelx: int, nely: int, 
         volfrac: float, penal: float, rmin: float, ft: int = 1,
         simulation_kw: Dict = {"grid": "regular",
                                "element order": 1,
                                "meshfile": None},
         nelz: Union[None,int] = None,
         filter_mode: str = "matrix",
         lin_solver: str = "scipy-direct", 
         preconditioner: Union[None,Callable,str] = None,
         assembly_mode: str = "full",
         materials_kw: Dict = {"E": 1.}, 
         body_forces_kw: Dict = {},
         bcs: Callable = mbb_2d, 
         lk: Union[None,Callable] = None, 
         l: Union[float,List,np.ndarray] = 1.,
         obj_func: Callable = compliance, 
         obj_kw: Dict = {},
         el_flags: Union[None,np.ndarray] = None,
         optimizer: str = "mma", 
         optimizer_kw: Union[None,Dict] = None,
         mix: Union[None,float] = None,
         accelerator_kw: Dict = {"accel_freq": 4,
                                 "accel_start": 20,
                                 "max_history": 0,
                                 "accelerator": None},
         nouteriter: int = 2000, ninneriter: int = 15,
         file: str = "topopt",
         display: bool = True, 
         export: bool = True,
         write_log: bool = True,
         debug: int = 0) -> Tuple[np.ndarray,float]:
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
    ft : int
        integer flag for the filter. 0 sensitivity filtering,
        1 density filtering, -1 no filter.
    nelz : int or None
        number of elements in z direction. If None, simulation is 2d.
    filter_mode : str
        indicates how filtering is done. Possible values are "matrix" or
        "helmholtz". If "matrix", then density/sensitivity filters are
        implemented via a sparse matrix and applied by multiplying
        said matrix with the densities/sensitivities.
    solver : str
        solver for linear systems. Check function lin solve for available
        options.
    preconditioner : str or None
        preconditioner for linear systems.
    assembly_mode : str
        whether full or only lower triangle of linear system / matrix is
        created.
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
    ninneriter: int
        number of inner iterations for GCMMA
    display : bool
        if True, plot design evolution to screen
    file : str
        name of output files
    export : bool
        if True, export design as vtk file.
    write_log : bool
        if True, write a log file and display results to command line.
    debug : bool
        if True, print extra output for debugging.

    Returns
    -------
    None.

    """
    #
    check_simulation_params(simulation_kw)
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if write_log:
        # check if log file exists and if True delete
        to_log = init_logging(logfile=file)
        #
        to_log(f"minimum compliance problem with optimizer {optimizer}")
        to_log(f"number of spatial dimensions: {ndim}")
        if ndim == 2:
            to_log(f"elements: {nelx} x {nely}")
        elif ndim == 3:
            to_log(f"elements: {nelx} x {nely} x {nelz}")
        if volfrac is not None:
            to_log(f"volfrac: {volfrac} rmin: {rmin}  penal: {penal}")
        else:
            to_log(f"rmin: {rmin}  penal: {penal}")
        to_log("filter: " + ["Sensitivity based",
                             "Density based",
                             "Haeviside Guest",
                             "Haeviside complement Sigmund 2007",
                             "Haeviside eta projection",
                             "Volume Preserving eta projection",
                             "No filter"][ft])
        to_log(f"filter mode: {filter_mode}")
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
    x = volfrac * np.ones( (n,1), dtype=float,order='F')
    #x = np.random.rand(n)
    #x = x/x.mean() * volfrac
    xTilde = x.copy()
    xPhys = x.copy()
    # initialize arrays for gradients
    dobj = np.zeros( x.shape,order="F")
    # initialize constraints
    n_constr = 0
    if volfrac is not None:
        n_constr += 1
    constrs = np.zeros( (n_constr, 1) )
    dconstrs = np.zeros( (n, n_constr) )
    # initialize solver
    if optimizer_kw is None:
        if optimizer in ["oc","ocm","ocg"]:
            # must be initialized to use the NGuyen/Paulino OC approach
            g = 0
            #
            max_history = 2
        elif optimizer == "mma":
            # mma needs results of the two previous iterations
            max_history = 3
            #
            if optimizer_kw is None:
                optimizer_kw = mma_defaultkws(x.shape[0],ft=ft,n_constr=1)
        elif optimizer == "gcmma":
            # gcmma needs results of the two previous iterations
            max_history = 3
            #
            if optimizer_kw is None:
                optimizer_kw = gcmma_defaultkws(x.shape[0],ft=ft,n_constr=1)
        else:
            raise ValueError("Unknown optimizer: ", optimizer)
    # handle element element flags
    if el_flags is not None and optimizer in ["mma","gcmma"]:
        # passive
        mask = el_flags == 1
        optimizer_kw["xmin"][mask] = 0.
        optimizer_kw["xmax"][mask] = 0.+1e-9
        x[mask,0] = 1.
        xPhys[mask,0] = 1.
        # active
        mask = el_flags == 2
        optimizer_kw["xmin"][mask] = 1.- 1e-9
        optimizer_kw["xmax"][mask] = 1.
        x[mask] = 1.
        xPhys[mask] = 1.
    # Max and min Young's modulus
    Emin = 1e-9
    E = 1.0
    # get element stiffness matrix
    KE = lk(l=l)
    if ndim == 2:
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        n_ndof = int(KE.shape[-1]/4)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*n_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=n_ndof)
    elif ndim == 3:
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        n_ndof = int(KE.shape[-1]/8)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*(nelz+1)*n_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,nelz=nelz,
                                                   nnode_dof=n_ndof)
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
            if ndim == 2 and n_ndof != 1:
                fixed = np.array([0,1,3])
            elif ndim == 3 and n_ndof != 1:
                fixed = np.array([0,1,2,4,5,7,8])
            elif n_ndof == 1:
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
            if ndim == 2 and n_ndof!=1:
                lf = lf_bodyforce_2d
            elif ndim == 3 and n_ndof!=1:
                lf = lf_bodyforce_3d
            fe_dens = lf_bodyforce_2d(b=body_forces_kw["density_coupled"])
        else:
            fe_dens = None
        #
        if len([key for key in body_forces_kw.keys() \
                if key not in ["density_coupled","strain_uniform"]]):
            raise NotImplementedError("One type of bodyforce/source has not yet been implemented.")
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat, mode=assembly_mode)
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
    if filter_mode == "convolution" and ndim == 2:
        invmapping = partial(map_imgtoel,
                             nelx=nelx,nely=nely)
    elif filter_mode == "convolution" and ndim == 3:
        invmapping = partial(map_voxeltoel,
                             nelx=nelx,nely=nely,nelz=nelz)
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                      rmin=rmin,ndim=ndim)
    elif filter_mode == "convolution":
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                           rmin=rmin,
                                           mapping=mapping,
                                           invmapping=invmapping)
        print(h.shape,hs.shape)
    elif filter_mode == "helmholtz" and ft in [0,1]:
        KF,TF = assemble_helmholtz_filter(nelx=nelx,nely=nely,nelz=nelz,
                                          rmin=rmin, l=l,
                                          n1=n1,n2=n2,n3=n3,n4=n4)
        # LU decomposition. returns a function for solving, not the matrices
        lu_solve = factorized(KF)
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
    if display:
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
    # initialize iteration history
    if max_history and accelerator_kw is None:
        xhist = [x.copy() for i in np.arange(max_history)]
    elif max_history >= accelerator_kw["max_history"]:
        xhist = [x.copy() for i in np.arange(max_history)]
    elif max_history < accelerator_kw["max_history"]:
        xhist = [x.copy() for i in np.arange(accelerator_kw["max_history"])]
    # optimization loop
    for loop in np.arange(nouteriter):
        # solve FEM, calculate obj. func. and gradients.
        # for
        if optimizer in ["oc","mma", "ocm","ocg"] or\
           (optimizer in ["gcmma"] and ninneriter==0) or\
           loop==0:
            # update physical properties of the elements and thus the entries
            # of the elements
            scale = simp(xPhys=xPhys,penal=penal,eps=1e-9)
            Kes = KE[None,:,:]*scale[:,:,None]
            if assembly_mode == "full":
                # this here is more memory efficient than Kes.flatten() as it
                # provides a view onto the original Kes array instead of a copy
                sK = Kes.reshape(np.prod(Kes.shape))
            elif assembly_mode == "lower":
                sK = Kes[:,assm_indcs[:,0],assm_indcs[:,1]].reshape( n*int(KE.shape[-1]/2*(KE.shape[-1]+1)))
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
            u[free, :], fact, precond = solve_lin(K=K, rhs=rhs[free],
                                                  solver=lin_solver,
                                                  preconditioner=preconditioner)
            # objective and sensitivities with regards to object
            obj = 0
            dobj[:] = 0.
            for i in np.arange(f.shape[1]):
                # obj. value, selfadjoint variables, self adjoint flag
                obj,rhs_adj,self_adj = obj_func(obj=obj, i=i,
                                                xPhys=xPhys,u=u,
                                                KE=KE, edofMat=edofMat,
                                                Kes=Kes,
                                                Amax=E,Amin=Emin,
                                                cellVolume=cellVolume,
                                                penal=penal,
                                                **obj_kw)
                # if problem not self adjoint, solve for adjoint variables and
                # calculate derivatives, else use analytical solution
                if self_adj:
                    #dobj[:] += rhs_adj
                    adj = np.zeros(f.shape)
                    adj[free] = rhs_adj[free]
                else:
                    adj = np.zeros(f.shape)
                    adj[free],_,_ = solve_lin(K, rhs=rhs_adj[free],
                                            solver=lin_solver, P=precond,
                                            preconditioner = preconditioner)
                # update sensitivity for quantities that need a small offset to
                # avoid degeneracy of the FE problem
                # standard contribution of element stiffness/conductivity
                dobj_offset = np.matvec(KE,u[edofMat,i])
                # contribution due to force induced by strain
                if "strain_uniform" in body_forces_kw.keys():
                    dobj_offset -= fe_strain[None,:,i]
                # generic density dependent element wise force
                if f0 is not None:
                    dobj_offset -= f0[None,:,i]
                #
                dobj[:,0] += (simp_dx(xPhys=xPhys, eps=1e-9, penal=penal)*\
                             adj[edofMat,i]*dobj_offset).sum(axis=1)
                # update sensitivity for quantities that do not need a small
                # offset to avoid degeneracy of the FE problem
                if "density_coupled" in body_forces_kw.keys():
                    dobj[:,0] -= simp_dx(xPhys=xPhys, eps=0., penal=1.)[:,0]*\
                                         np.dot(adj[edofMat,i],fe_dens[:,i])
                if debug:
                    print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                           loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        # optimizer is unknown.
        else:
            raise NotImplementedError("Unknown optimizer.")
        # Constraints and constraint gradients
        constrs[:,0] = 0.
        dconstrs[:,:] = 0.
        if volfrac is not None:
            constrs[0,0] = xPhys.mean() - volfrac
            if optimizer in ["mma","gcmma"]:
                dconstrs[:,0:1] = np.ones(x.shape) /(x.shape[0]*volfrac)
            elif optimizer in ["oc","ocm","ocg"]:
                dconstrs[:,0] = np.ones(x.shape[0])
        if debug:
            print("Pre-Sensitivity Filter: it.: {0}, dobj: {1:.10f}, dv: {2:.10f}".format(
                   loop,
                   np.max(dobj),
                   np.min(dconstrs)))
        # Sensitivity filtering:
        if ft == 0 and filter_mode == "matrix":
            dobj[:] = np.asarray(H@(x*dobj) /
                                 Hs) / np.maximum(0.001, x)
            #dobj[:] = H @ (dc*x) / Hs / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "convolution":
            dobj[:] = invmapping( convolve(mapping(dobj),
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
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, max. dobj: {1:.10f}, min. dv: {2:.10f}".format(
                   loop,
                   np.max(dobj),
                   np.min(dconstrs)))
        # density update by optimizer
        # optimality criteria
        if optimizer=="oc":
            (x[:,0], g) = oc_top88(x=x[:,0], volfrac=volfrac,
                                 dc=dobj[:,0], dv=dconstrs[:,0], g=g,
                                 el_flags=el_flags)
        elif optimizer=="ocm":
            (x[:,0], g) = oc_mechanism(x=x[:,0], volfrac=volfrac,
                                     dc=dobj[:,0], dv=dconstrs[:,0], g=g,
                                     el_flags=el_flags)
        elif optimizer=="ocg":
            (x[:,0], g) = oc_generalized(x=x[:,0], volfrac=volfrac,
                                       dc=dobj[:,0], dv=dconstrs[:,0], g=g,
                                       el_flags=el_flags)
        # method of moving asymptotes
        elif optimizer=="mma":
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                                 n=x.shape[0],
                                                                 iter=i,
                                                                 xval=x,
                                                                 xold1=xhist[-1],
                                                                 xold2=xhist[-2],
                                                                 f0val=obj,
                                                                 df0dx=dobj,
                                                                 fval=constrs,
                                                                 dfdx=dconstrs.T,
                                                                 **optimizer_kw)

            # update asymptotes
            optimizer_kw["low"] = low
            optimizer_kw["upp"] = upp
            x = xmma.copy()
        if debug:
            print("Post Density Update: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # mixing
        if ((loop-accelerator_kw["accel_start"])%accelerator_kw["accel_freq"])==0 \
            and loop >= accelerator_kw["accel_start"] and \
            accelerator_kw["accelerator"] is not None:
            x[:] = accelerator_kw["accelerator"](x=x.reshape( np.prod(x.shape), order="F" ),
                                                 xhist=[_x.reshape( np.prod(x.shape), order="F" ) for _x in xhist],
                                                 **accelerator_kw).reshape(x.shape,order="F")
        elif mix is not None:
            x[:] = xhist[-1]*(1-mix) + x*mix
        # append history
        xhist.append(x.copy())
        # prune history if too long
        if len(xhist)> max_history+1:
            xhist = xhist[-max_history-1:]
        if debug:
            print("Post Mixing Update: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Filter design variables
        if ft == 0:
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
        if debug:
            print("Post Density Filter: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Compute the change by the inf. norm
        change = np.abs(xhist[-1] - xhist[-2]).max()
        # Plot to screen
        if display:
            if ndim == 2:
                plotfunc(mapping(-xPhys))
            elif ndim == 3:
                im.set_array(mapping(-xPhys))
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if write_log:
            to_log("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch.: {3:.10f}".format(
                         loop+1, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break
    #
    if display:
        plt.show()
        input("Press any key...")
    #
    if export:
        export_vtk(filename=file,
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,x=x,
                   u=u,f=f,volfrac=volfrac)
    return x, obj
