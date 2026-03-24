# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict
from functools import partial
import logging
#
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.ndimage import convolve
#
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# functions to create filters
from topoptlab.filter.matrix_filter import assemble_matrix_filter
from topoptlab.filter.haeviside_projection import find_eta
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import Lbracket
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.strain_measures import infini_strain_matrix
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
from topoptlab.elements.stress_measures import von_mises_stress
from topoptlab.elements.stress_measures import dsvm_ds
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d,isotropic_3d
from topoptlab.elements.linear_elasticity_2d import _lf_strain_2d,lk_linear_elast_aniso_2d
from topoptlab.elements.linear_elasticity_3d import _lf_strain_3d,lk_linear_elast_aniso_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d

from topoptlab.material_interpolation import simp,simp_dx,ramp,ramp_dx
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# constrained optimizers
from topoptlab.optimizer.optimality_criterion import oc_top88,oc_mechanism,oc_generalized
from topoptlab.optimizer.mma_utils import update_mma,mma_defaultkws,gcmma_defaultkws
from topoptlab.objectives import stress_pnorm
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk,threshold
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
# logging related stuff
from topoptlab.log_utils import EmptyLogger,init_logging
# drawing function
from topoptlab.draw_functions import spring, hinged_support

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft,
         Emax=1, nu=0.3,
         filter_mode="matrix",
         lin_solver="cvxopt-cholmod", preconditioner=None,
         assembly_mode="full",
         body_forces_kw={},
         bcs=Lbracket, l=1.,
         obj_func=stress_pnorm, obj_kw={},
         el_flags=None,
         optimizer="mma", optimizer_kw = None,
         Pnorm=10,
         penal_sig=0.5,             
         nouteriter=2000,ninneriter=15,
         file="lbracket",
         matinterpol: Callable = ramp,
         matinterpol_dx: Callable = ramp_dx,
         matinterpol_kw: Dict = {"eps": 1e-9, "penal": 3.},
         display=False,export=False,write_log=False,
         debug=0):
    """
    Run topology optimization for stress minimization 
    using an pnorm relaxed von Mises stress objective.

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
        returns the boundary conditions.
    l : float or tuple of length (ndim) or np.ndarray of shape (ndim)
        lengths of each element
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
    nouteriter: int
        number of TO iterations
    ninneriter: int
        number of iterations for GCMMA
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
    x : ndarray of shape (n, 1)
        Final design variable field.

    obj : float
        Final objective value evaluated on the thresholded design.

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    if write_log:
        # check if log file exists and if True delete
        to_log = init_logging(logfile=file)
        #
        to_log(f"a Lbracket to validate stress minimization with optimizer {optimizer}")
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
    else:
        log = EmptyLogger()
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
    elif ndim == 3:
        n = nelx * nely * nelz
    #
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones((n,1), dtype=float,order='F')
    xold = x.copy()
    xTilde = x.copy()
    xPhys = x.copy()
    xBase = x.copy()
    if ft == 5:
        beta = 1
        eta = find_eta(eta0=0.5, xTilde=xTilde, beta=beta, volfrac=volfrac)
    else:
        beta = None
    #
    if ndim == 2:
        xe = l*np.array([[[-1.,-1.],
                          [1.,-1.],
                          [1.,1.],
                          [-1.,1.]]])/2 * np.ones((n,1,1))
        xi  = np.array([0.0])
        etaa = np.array([0.0])
        B = infini_strain_matrix(xi=xi, eta=etaa, xe=xe, zeta= None, all_elems=True, shape_functions_dxi=shape_functions_dxi)
    elif ndim == 3:
        xe = l*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])/2 \
            * np.ones((n,1,1))
        xi  = np.array([0.0])
        etaa = np.array([0.0])
        zeta = np.array([0.0])
        B = infini_strain_matrix(xi=xi, eta=etaa, zeta=zeta, xe=xe, all_elems=True, shape_functions_dxi=shape_functions_dxi)
    if ndim ==2:
        # stiffness tensor
        cs = [isotropic_2d(E=Emax, nu=nu) \
              for i in np.arange(int(nely))]
        cs = np.tile(np.stack(cs),(nelx,1,1))
    if ndim ==3:
        # stiffness tensor
        cs = [isotropic_3d(E=Emax, nu=nu) \
              for i in np.arange(int(nely))]
        cs = np.tile(np.stack(cs),(nelx*nelz,1,1))
    # initialize arrays for gradients
    dobj = np.zeros((n, 1),order="F")
    dv = np.ones((n, 1),order="F")
    # initialize solver
    # upper and lower volume constr 
    n_constr = 2
    if optimizer_kw is None:
        if optimizer in ["oc","ocm","ocg"]:
            # must be initialized to use the NGuyen/Paulino OC approach
            g = 0
        elif optimizer == "mma":
            # mma needs results of the two previous iterations
            nhistory = 2            
            # n variables: x only
            xhist = [x.copy(), x.copy()]
            nvars = n
            optimizer_kw = mma_defaultkws(nvars, ft=ft, n_constr=n_constr)
            optimizer_kw["xmin"] = np.zeros(nvars)
            optimizer_kw["xmax"] = np.ones(nvars)
            optimizer_kw["low"]  = np.ones(nvars)
            optimizer_kw["upp"]  = np.ones(nvars)           
            if ft == 5:
                optimizer_kw["move"] = 0.05
        else:
            raise ValueError("Unknown optimizer: ", optimizer)
    # handle element element flags
    if el_flags is not None and optimizer in ["mma","gcmma"]:
        # passive
        mask = el_flags == 1
        optimizer_kw["xmin"][:n][mask] = 0.
        optimizer_kw["xmax"][:n][mask] = 0.+1e-9
        x[mask] = 0.0
        xPhys[mask] = 0.0
        # active
        mask = el_flags == 2
        optimizer_kw["xmin"][:n][mask] = 1.-1e-9
        optimizer_kw["xmax"][:n][mask] = 1.
        x[mask] = 1.
        xPhys[mask] = 1.
    # get element matrices
    if ndim == 2:
        KE  = np.zeros((n, 8, 8), dtype=float)
        for e in np.arange(n):
            KE[e, :, :] = lk_linear_elast_aniso_2d(c=cs[e, :, :], l=l,g=np.array([0.]),t=1.0)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3 dimension
        n_ndof = int(KE.shape[-1]/4)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*n_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=n_ndof)
    elif ndim == 3:
        KE  = np.zeros((n, 24, 24), dtype=float)
        for e in np.arange(n):
            KE[e, :, :] = lk_linear_elast_aniso_3d(c=cs[e, :, :], l=l,g=np.array([0.,0.]))
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
                lf = _lf_strain_2d
            elif ndim == 3:
                lf = _lf_strain_3d
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
            u0 = None
        #
        if "density_coupled" in body_forces_kw.keys():
            # fetch functions to create body force
            if ndim == 2 and n_ndof!=1:
                lf = lf_bodyforce_2d
            elif ndim == 3 and n_ndof!=1:
                lf = lf_bodyforce_3d
            fe_dens = lf(b=body_forces_kw["density_coupled"],l=l)
        else:
            fe_dens = None
        #
        if len([key for key in body_forces_kw.keys() \
                if key not in ["density_coupled","strain_uniform"]]):
            raise NotImplementedError("One type of bodyforce/source has not yet been implemented.")
    
    
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat,mode=assembly_mode)
    if assembly_mode == "lower":
        assm_indcs = np.column_stack(np.triu_indices_from(KE))
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
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx,nely=nely,nelz=nelz,
                                 ndof=ndof)
    f0 = None
    if display:
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        if ndim == 2:
            fig,ax = plt.subplots(1,1)
            im = ax.imshow(mapping(-xPhys), cmap='gray',
                           interpolation='none', norm=Normalize(vmin=-1, vmax=0))
            plotfunc = im.set_array
        elif ndim == 3:
            raise NotImplementedError("Plotting in 3D not yet implemented.")
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        ax.axis("off")
        fig.show()
    # optimization loop
    loopbeta = 0
    for loop in np.arange(nouteriter):
        #
        loopbeta += 1  
        # calculate / interpolate material properties
        dscale = matinterpol_dx(xPhys=xPhys, **matinterpol_kw)
        scale = matinterpol(xPhys=xPhys,**matinterpol_kw)
        Kes = KE*scale[:,:,None]
        # solve FEM, calculate obj. func. and gradients.
        # for
        if optimizer in ["oc","mma", "ocm","ocg"] or\
           (optimizer in ["gcmma"] and ninneriter==0) or\
           loop==0:
            # update physical properties of the elements and thus the entries
            # of the elements
            if assembly_mode == "full":
                # this here is more memory efficient than Kes.flatten() as it
                # provides a view onto the original Kes array instead of a copy
                sK = Kes.reshape(np.prod(Kes.shape))
            elif assembly_mode == "symmetry":
                sK = Kes[:,assm_indcs[0],assm_indcs[1]].reshape( n*ndof*(ndof+1) )
            # Setup and solve FE problem
            # assemble system matrix
            K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                                ndof=ndof,solver=lin_solver,
                                springs=springs)
            # assemble forces due to body forces
            f_body = np.zeros(f.shape)
            for bodyforce in body_forces_kw.keys():
                # assume each strain is a column vector in Voigt notation
                if "strain_uniform" in body_forces_kw.keys():
                    fes = fe_strain[None,:,:]*scale[:,:,None]
                    np.add.at(f_body,
                              edofMat,
                              fes)
                if "density_coupled" in body_forces_kw.keys():
                    fes = fe_dens[None,:,:]*ramp(xPhys=xPhys, eps=0., penal=0.)[:,:,None]
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
            # Objective and objective gradient
            obj = 0
            dobj[:] = 0.
            for i in np.arange(f.shape[1]):

                ue = u[edofMat, i]
                # interpolated constitutive tensor
                C_es = scale[:, :, None] * cs
                # element strain
                strain = np.einsum('eij,ej->ei', B, ue, optimize=True)
                # element stress
                stress = np.einsum('eij,ej->ei', C_es, strain, optimize=True)
                # von Mises stress and derivative
                stress_vm = von_mises_stress(stress=stress, ndim=ndim)
                dsvm = dsvm_ds(stress=stress,stress_vm=stress_vm,ndim=ndim)
                obj, rhs_adj, self_adj = obj_func(
                    u=u,
                    i=i,              
                    edofMat=edofMat,
                    B=B,
                    C_es=C_es,
                    xPhys=xPhys,
                    stress_vm=stress_vm,
                    dsvm=dsvm,
                    penal_sig=penal_sig,
                    Pnorm=Pnorm,
                    obj=obj,
                    **obj_kw
                )
                # update sensitivity for quantities that need a small offset to
                # avoid degeneracy of the FE problem
                #"""
                # if problem not self adjoint, solve for adjoint variables and
                # calculate derivatives, else use analytical solution
                # if problem not self adjoint, solve for adjoint variables and
                # calculate derivatives, else use analytical solution
                if self_adj:
                    #dobj[:] += rhs_adj
                    lamU = np.zeros(f.shape)
                    lamU[free] = rhs_adj[free]
                else:
                    lamU = np.zeros(f.shape)
                    lamU[free],_,_ = solve_lin(K, rhs=rhs_adj[free],
                                            solver=lin_solver, P=precond,
                                            preconditioner = preconditioner)
                # update sensitivity for quantities that need a small offset to
                # avoid degeneracy of the FE problem
                # standard contribution of element stiffness/conductivity
                dobj_offset = np.matvec(KE,u[edofMat,i])
                # thermal expansion
                # dobj_offset[:] -= fTe[:,:,0]
                # contribution due to force induced by strain
                if "strain_uniform" in body_forces_kw.keys():
                    dobj_offset -= fe_strain[None,:,i]
                #
                if f0 is not None:
                    dobj_offset -= f0[None,:,i]
                dobj[:,0] += (dscale*lamU[edofMat,i]*dobj_offset).sum(axis=1)
                # update sensitivity for quantities that do not need a small
                # offset to avoid degeneracy of the FE problem
                if "density_coupled" in body_forces_kw.keys():
                    dobj[:,0] -= ramp_dx(xPhys=xPhys, eps=0., penal=0.)[:,0]*\
                                    np.dot(lamU[edofMat,i],fe_dens[:,i]) 
                if debug:
                    print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                           loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        
        if loop == 0:
            if export:
                export_vtk(
                    filename=f"{file}_it{loop+1:04d}",
                    nelx=nelx, nely=nely, nelz=nelz,
                    xPhys=xPhys, x=x,
                    elem_size=l,
                    stress_vm=stress_vm,
                    u=u, f=f+f_body,
                    volfrac=volfrac)
    
        # Build constraint values 
        constr_list = []
        vol_up = xPhys.mean() - volfrac - 1e-5
        vol_lo = volfrac - xPhys.mean() - 1e-5
        constr_list.extend([vol_up, vol_lo])
        constrs = np.asarray(constr_list, dtype=float).ravel()

        dconstr = np.zeros((n, n_constr), dtype=float)
        col = 0
        dconstr[:n, col] =  1.0 / n; col += 1      # vol upper
        dconstr[:n, col] = -1.0 / n; col += 1      # vol lower
        # Sensitivity filtering:
        if ft == 0 and filter_mode == "matrix":
            dobj[:] = np.asarray(H@(x*dobj) /
                                 Hs) / np.maximum(0.001, x)
            #dobj[:] = H @ (dc*x) / Hs / np.maximum(0.001, x)

        elif ft == 1 and filter_mode == "matrix":
            dobj[:] = np.asarray(H*(dobj/Hs))
            for j in range(col):
                dconstr[:n, j] = np.asarray(H*((dconstr[:n, j])[:, None] / Hs))[:, 0]
        elif ft == 5:
            xTilde[:] = np.asarray(H * x/ Hs)
            xBase[:] = xTilde.copy()
            dx = beta * (1 - np.tanh(beta * (xBase - eta))**2) /\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))   
            dobj[:] = dobj * dx   # dJ/d(xP_input)
            dobj[:] = np.asarray(H * (dobj / Hs))
            for j in range(col):
                dconstr[:n, j] = dconstr[:n, j] * dx[:, 0]  # dgj/d(xBase)
                dconstr[:n, j] = np.asarray(H * ((dconstr[:n, j])[:, None] / Hs))[:, 0]
        elif ft == -1:
            pass
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, max. dobj: {1:.10f}, min. dv: {2:.10f}".format(
                   loop,
                   np.max(dobj),
                   np.min(dv)))
        # density update by solver
        xold[:] = x
        xPhysold = xPhys.copy()
        # optimality criteria
        if optimizer=="oc":
            (x[:], g) = oc_top88(x=x, volfrac=volfrac,
                                 dc=dobj, dv=dv, g=g,
                                 el_flags=el_flags)
        elif optimizer=="ocm":
            (x[:], g) = oc_mechanism(x=x, volfrac=volfrac,
                                     dc=dobj, dv=dv, g=g,
                                     el_flags=el_flags)
        elif optimizer=="ocg":
            (x[:], g) = oc_generalized(x=x, volfrac=volfrac,
                                       dc=dobj, dv=dv, g=g,
                                       el_flags=el_flags)
        # method of moving asymptotes
        elif optimizer == "mma":
            nvars = n
            xold1  = np.asarray(xhist[-1], dtype=float).reshape(nvars, 1)  
            xold2  = np.asarray(xhist[-2], dtype=float).reshape(nvars, 1)  
            for j in ("xmin", "xmax", "low", "upp"):
                optimizer_kw[j] = np.asarray(optimizer_kw[j], dtype=float).reshape(nvars, 1)  
            xmma, ymma, zmma, lam, xsi, eta_mma, mu, zet, s, low, upp = update_mma(
                x=np.asarray(x, dtype=float).ravel(), xold1=xold1, xold2=xold2, 
                xPhys=np.asarray(x, dtype=float).ravel(),
                obj=np.asarray(obj, dtype=float), dobj=np.asarray(dobj, dtype=float).ravel(),
                constrs=constrs,
                dconstr=np.asarray(dconstr, dtype=float),
                iteration=loop, **optimizer_kw
            )
            x[:] = np.asarray(xmma, dtype=float).reshape(nvars, 1) 
            xhist.pop(0); xhist.append(x.copy())
            optimizer_kw["low"] = low; optimizer_kw["upp"] = upp
            # print(f"fval: {constrs}, max_violation: {np.maximum(constrs,0).max():.3e}")    

        if debug:
            print("Post Density Update: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1 and filter_mode == "matrix":
            xTilde[:] = np.asarray(H*x/Hs)
            xPhys[:] = xTilde         
            #xPhys[:] = H @ x / Hs
        elif ft in [5] and filter_mode == "matrix":
            xTilde[:] = np.asarray(H*x/Hs)
            xBase = xTilde.copy()
            eta = find_eta(eta0=eta, xTilde=xBase, beta=beta, volfrac = volfrac)
            xPhys[:] = (np.tanh(beta*eta)+np.tanh(beta * (xBase - eta)))/\
                       (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        elif ft == -1:
            xPhys[:]  = x
        if debug:
            print("Post Density Filter: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        
        # Compute the change by the inf. norm
        change = np.abs(xPhys-xPhysold).max()
        export_every = 100
        if export and ((loop + 1) % export_every == 0):
            export_vtk(
                filename=f"{file}_it{loop+1:04d}",
                nelx=nelx, nely=nely, nelz=nelz,
                xPhys=xPhys, x=x,
                elem_size=l,
                stress_vm=stress_vm,
                u=u, f=f+f_body,
                volfrac=volfrac)

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
            to_log("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch: {3:.10f}".format(
                         loop+1, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01 and beta is None and loop >1000:
            break
        # parameter continuation for beta in volume projection
        elif (ft == 5) and (beta < 256) and \
            (loopbeta >= 100 or change < 0.01):
            beta = 1.5 * beta
            loopbeta = 0
            if write_log:
                to_log(f"Parameter beta increased to {beta}")
        elif (ft == 5) and (beta >= 256) and (change < 0.01):
            break
    #
    if display:
        plt.show()
        input("Press any key...")
    #
    xThresh = threshold(xPhys,volfrac)
    scale = matinterpol(xPhys=xThresh,eps=1e-9, penal=penal)
    # update physical properties of the elements and thus the entries
    # of the elements 
    if assembly_mode == "full":
        sK = (scale[:,:,None] * KE).flatten() 
    # Setup and solve FE problem
    # To Do: loop over boundary conditions if incompatible
    # assemble system matrix
    K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                        ndof=ndof,solver=lin_solver,
                        springs=springs)
    u0 = None
    f_body = np.zeros(f.shape)
    for bodyforce in body_forces_kw.keys():
        # assume each strain is a column vector in Voigt notation
        if "strain_uniform" in body_forces_kw.keys():
            fes = fe_strain[None,:,:]*scale[:,:,None]
            np.add.at(f_body,
                      edofMat,
                      fes)
        if "density_coupled" in body_forces_kw.keys():
            fes = fe_dens[None,:,:]*ramp(xPhys=xThresh, eps=0., penal=0.)[:,:,None]
            np.add.at(f_body,
                      edofMat,
                      fes)
    # assemble right hand side
    rhs = f+f_body
    # apply boundary conditions to matrix
    K = apply_bc(K=K,solver=lin_solver,
                 free=free,fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    u_bw = np.zeros(u.shape)
    u_bw[free, :], fact, precond = solve_lin(K=K, rhs=rhs[free],
                                          solver=lin_solver,
                                          preconditioner=preconditioner)
    i = 0
    C_es = scale[:, :, None] * cs
    ue = u_bw[edofMat, i]
    # element strain
    strain = np.einsum('eij,ej->ei', B, ue, optimize=True)
    # element stress
    stress = np.einsum('eij,ej->ei', C_es, strain, optimize=True)
    # von Mises stress
    stress_vm = von_mises_stress(stress=stress, ndim=ndim)
    dsvm = dsvm_ds(stress=stress,stress_vm=stress_vm,ndim=ndim)
    obj=0.                  
    obj, rhs_adj, self_adj = obj_func(
                    u=u_bw,
                    i=0,              
                    edofMat=edofMat,
                    B=B,
                    C_es=C_es,
                    xPhys=xThresh,
                    stress_vm=stress_vm,
                    dsvm=dsvm,
                    penal_sig=penal_sig,
                    Pnorm=Pnorm,
                    obj=obj,
                    **obj_kw
                )
    #
    if write_log:
        to_log("final.: obj.: {0:.10f} vol.: {1:.10f}".format(obj, xThresh.mean()))
    #
    if export:
        export_vtk(
            filename=f"{file}_it{loop+1:04d}",
            nelx=nelx, nely=nely, nelz=nelz,
            xPhys=xPhys, x=x,
            elem_size=l,
            stress_vm=stress_vm,
            u=u_bw, f=f+f_body,
            volfrac=volfrac) 
    return u_bw, rhs_adj

def sketch(save=False):
    """
    Just a sketch to indicate boundary conditions etc.
    """

    fig,ax = plt.subplots(figsize=(16,6))
    # bottom layer
    ax.plot(np.array([0,1,1,0,0]),np.array([0,0,1,1,0]),c="k")
    # top layer
    ax.plot(np.array([0,1,1,0,0]),np.array([1,1,2,2,1]),c="k")
    # arrows for indication of layer orientation
    for i in range(5):
        ax.arrow(x=0.1 + i*0.2, y=0.7, dx=0., dy=-0.4,
                 width=0.002, head_length=0.2,
                 color="k")
        ax.arrow(x=0.05+ i*0.2, y=1.5, dx=0.1, dy=0,
                 width=0.01, color="k")
    # spring
    x,y = spring(x0=1.,y0=2.,
                 num_coils = 3, coil_width = 0.02,
                 coil_length = 0.4, points_per_coil = 8)
    ax.plot(x,y,
            linewidth=2.,color="gray")
    x,y = spring(x0=0.,y0=2.,
                 num_coils = 3, coil_width = 0.02,
                 coil_length = 0.4, points_per_coil = 8)
    ax.plot(x,y,
            linewidth=2.,color="gray")
    # mirror axis
    ax.axvline(x=0.5, ymin = -1, ymax = 3.,
               color="b", linestyle="--", linewidth=3.,
               alpha = 0.7)
    #
    hinged_support(x0=0.5,y0=0.,
                   ax=ax,fig=fig,
                   triangle_width=.15,
                   radius=0.05)
    #
    ax.axis("off")
    ax.set_ylim(-0.5,2.5)
    if save:
        plt.savefig(fname="sketch.pdf",format="pdf",bbox_inches="tight")
    plt.show()
    return

# The real main driver
if __name__ == "__main__":
    #
    #sketch(save=True)
    # Default input parameters
    nelx=100
    nely=nelx
    nelz=None
    volfrac=0.3
    rmin=0.05*nelx
    penal=3
    ft=5 # ft==0 -> sens, ft==1 -> dens
    elem_size=1.0
    nouteriter=2000
    export=True
    write_log=True
    display=True
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    if len(sys.argv)>7: nouteriter =int(sys.argv[7])
    if len(sys.argv) > 8: export = bool(int(sys.argv[8]))
    if len(sys.argv) > 9: write_log = bool(int(sys.argv[9]))
    if len(sys.argv) > 10: display = bool(int(sys.argv[10]))
    #
    if nelz is None:
        bcs=Lbracket
    else:
        raise ValueError("Only for 2D validation")
    # assume nelx = nely = 100
    el_flags = np.zeros(nelx*nely, dtype=np.int32)
    xs = np.arange(41, nelx, dtype=int)   
    ys = np.arange(0, 60, dtype=int)   
    idx = (xs[:, None] * nely+ ys[None, :]).astype(int).ravel()
    el_flags[idx] = 1

    u_bw, rhs_adj = main(nelx=nelx,nely=nely,volfrac=volfrac,penal=penal,rmin=rmin,ft=ft,
         obj_func=stress_pnorm,
         body_forces_kw={"density_coupled": np.array([0,-1e-7])},
         el_flags = el_flags,
         display=display,
         bcs=bcs,
         file='aLbracket',
         nouteriter=nouteriter,
         export=export,write_log=write_log)
    # for tests
    # np.savetxt("stress_lbracket_u_bw.csv", u_bw, delimiter=",")
    # np.savetxt("stress_lbracket_rhs_adj.csv", rhs_adj, delimiter=",")
    

