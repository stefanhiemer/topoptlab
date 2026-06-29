# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
# functions to create filters
from topoptlab.filter.kernels import hat_kernel
from topoptlab.filter.matrix_filter import assemble_matrix_filter
from topoptlab.filter.haeviside_projection import find_eta
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import Lbracket,mbb_2d
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
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_aniso_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_aniso_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d

from topoptlab.material_interpolation import simp,simp_dx,ramp,ramp_dx
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# constrained optimizers
from topoptlab.optimizer.mma_utils import update_mma,mma_defaultkws,gcmma_defaultkws,mmasub
from topoptlab.objectives import stress_pnorm,compliance
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk,threshold
# logging related stuff
from topoptlab.log_utils import EmptyLogger,init_logging

# MAIN DRIVER
def main(nelx: int, nely: int, nelz: int | None,
        volfrac: float, 
        penal: float, 
        rmin: float, 
        ft: int,
        Emax: float = 1.0, 
        nu: float = 0.3,
        filter_mode: str = "matrix", 
        lin_solver: str = "cvxopt-cholmod",
        preconditioner: str | None = None,
        assembly_mode: str = "full", 
        body_forces_kw: Dict | None = None,
        bcs: Callable = Lbracket, 
        l: float | np.ndarray = 1.0,
        obj_func: Callable = stress_pnorm, 
        obj_kw={},         
        el_flags: np.ndarray | None = None,
        optimizer_kw: Dict | None = None,
        Pnorm: float = 10, penal_sig: float = 0.5,
        use_stress_constraint: bool = False,
        stress_allow: None | float = 0.45,
        nouteriter: int = 2000, ninneriter: int = 15,
        file: str = "lbracket",
        matinterpol: Callable = simp, matinterpol_dx: Callable = simp_dx,
        matinterpol_kw: Dict = {"eps": 1e-9, "penal": 3.},
        display: bool = False, export: bool = False,
        write_log: bool = False,
        debug: int = 0) -> float:
    """
    Run topology optimization with compliance minimization under 
    stress constraint using an pnorm relaxed von Mises stress objective.
    Details please refer to: 
    Le, Chau, et al. "Stress-based topology optimization for continua." 
    Structural and Multidisciplinary Optimization 41.4 (2010): 605-620.
    One can also use mbb beam to validate the stress constr effect. 
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        Number of elements in the z direction. If ``None``, a 2D problem is
        solved. Otherwise, a 3D problem is assumed.
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
    Emax : float, optional
        Young's modulus of the solid material.
    nu : float, optional
        Poisson ratio.
    filter_mode : str, optional
        Filter implementation. Typically ``"matrix"``.
    lin_solver : str, optional
        Linear solver used for the finite element system.
    preconditioner : str or None, optional
        Preconditioner passed to the linear solver.
    assembly_mode : str, optional
        Matrix assembly mode. Usually ``"full"``. Other options depend on the
        FEM backend implementation.
    body_forces_kw : dict, optional
        Dictionary describing additional element load contributions. Supported
        keys include:
        - ``"strain_uniform"`` for strain-induced loads
        - ``"density_coupled"`` for density-dependent body forces
    bcs : callable, optional
        Boundary condition generator. It must return displacement array,
        external force array, fixed dofs, free dofs, and spring data.
    l : float or array_like, optional
        Element size. A scalar is broadcast to all spatial directions.
    obj_func : callable, optional
        Objective function callback. It must return the updated objective value,
        adjoint right-hand side, and a flag indicating whether the objective is
        self-adjoint.
    obj_kw : dict, optional
        Additional keyword arguments passed to ``obj_func``.
    el_flags : np.ndarray or None, optional
        Element activity flags.
    optimizer : str, optional
        Optimization algorithm. Supported values are ``"oc"``, ``"ocm"``,
        ``"ocg"``, ``"mma"``, and possibly ``"gcmma"`` if enabled in the
        surrounding implementation.
    optimizer_kw : dict or None, optional
        Additional optimizer settings. If ``None``, default values are created.
    Pnorm : float, optional
        Exponent used in the aggregated stress p-norm constraint.
    penal_sig : float, optional
        Stress relaxation exponent used in the stress aggregation model.
    use_stress_constraint : bool, optional
        If True, add a global stress constraint based on the relaxed p-norm
        of the element von Mises stress.
    stress_allow : float, optional
        Allowable stress used to normalize the stress constraint.
    nouteriter : int, optional
        Maximum number of outer topology optimization iterations.
    ninneriter : int, optional
        Number of inner iterations for algorithms that require them, such as
        GCMMA.
    file : str, optional
        Base name used for log and export files.
    matinterpol : callable, optional
        Material interpolation law that maps physical densities to stiffness
        scaling factors.
    matinterpol_dx : callable, optional
        Derivative of ``matinterpol`` with respect to the physical density.
    matinterpol_kw : dict, optional
        Additional keyword arguments passed to ``matinterpol`` and
        ``matinterpol_dx``.
    display : bool, optional
        If True, show the design evolution during optimization.
    export : bool, optional
        If True, export intermediate and final results to VTK.
    write_log : bool, optional
        If True, write iteration history to a log file and print progress.
    debug : int or bool, optional
        Debug flag. If enabled, print additional diagnostic information.
    Returns
    -------
    obj : float
        Final objective value evaluated on the thresholded design.
    """
    optimizer="mma"
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
        xe = np.array([[[-1.,-1.],
                        [1.,-1.],
                        [1.,1.],
                        [-1.,1.]]])/2
        from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
    else:
        ndim = 3
        create_edofMat = create_edofMat3d 
        xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])/2  
        from topoptlab.elements.trilinear_hexahedral import shape_functions_dxi 
    # total number of design variables/elements
    n = np.prod(np.array([nelx,nely,nelz])[:ndim])
    #
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    #
    xe = l*xe*np.ones((n,1,1))
    
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
        to_log = EmptyLogger()

    if body_forces_kw is None:
        body_forces_kw = {}

    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones((n,1), dtype=float,order='F')
    xold, xPhys, xTilde, xBase = x.copy(), x.copy(), x.copy(), x.copy()

    xi,eeta,zeta = ndim*[np.array([0.])] + int(3-ndim)*[None]
    B = infini_strain_matrix(xi=xi, 
                             eta=eeta, 
                             zeta=zeta, 
                             xe=xe, 
                             all_elems=True, 
                             shape_functions_dxi=shape_functions_dxi)
    if ft == 5:
        beta = 1
        eta = find_eta(eta0=0.5, xTilde=xTilde, beta=beta, volfrac=volfrac)
    else:
        beta = None
    #
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
    # upper, lower volume constr or stress constraint
    n_constr = 2 + int(use_stress_constraint)  

    if optimizer == "mma":
        # mma needs results of the two previous iterations
        nhistory = 3            
        # n variables: x only
        xhist = [x.copy(), x.copy()]
        nvars = n
        optimizer_kw = mma_defaultkws(nvars, ft=ft, n_constr=n_constr) 
        if ft == 5:
            optimizer_kw["move"] = 0.05
    # handle element element flags
    if el_flags is not None:
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
    KE  = np.zeros((n, ndim*2**ndim, ndim*2**ndim), 
                   dtype=float)
    if ndim == 2:
        for e in np.arange(n):
            KE[e, :, :] = lk_linear_elast_aniso_2d(c=cs[e, :, :], 
                                                   l=l,
                                                   g=np.array([0.]),
                                                   t=1.0)
    elif ndim == 3:
        for e in np.arange(n):
            KE[e, :, :] = lk_linear_elast_aniso_3d(c=cs[e, :, :], l=l,g=np.array([0.,0.]))
    # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
    n_ndof = int(KE.shape[-1]/2**ndim)
    # number of degrees of freedom
    ndof = (nelx+1)*(nely+1)*n_ndof
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,
                                             nely=nely,
                                             nelz=nelz,
                                             nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat,
                              mode="full")
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,
                                      nely=nely,
                                      nelz=nelz,
                                      rmin=rmin,
                                      ndim=ndim, 
                                      kernel_fn=hat_kernel)
    else:
        raise ValueError("this tutorial only permits filter_mode 'matrix'.")

    # fetch body forces
    if "density_coupled" in body_forces_kw:
        # fetch functions to create body force
        if ndim == 2 and n_ndof!=1:
            lf = lf_bodyforce_2d
        elif ndim == 3 and n_ndof!=1:
            lf = lf_bodyforce_3d
        fe_dens = lf(b=body_forces_kw["density_coupled"],l=l)
    else:
        fe_dens = None
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx, 
                                 nely=nely,
                                 nelz=nelz,
                                 ndof=ndof)
    if display:
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        fig,ax = plt.subplots()
        im = ax.imshow(-xPhys.reshape((nely,nelx),order="F"), cmap='gray',
                       interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        fig.show()
    dconstr_stress_dx = np.zeros((n, 1), order="F")
    constr_stress = 0.0
    c_stress = None
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
        # update physical properties of the elements and thus the entries
        # of the elements
        sK = Kes.reshape(np.prod(Kes.shape))
        # Setup and solve FE problem
        # assemble system matrix
        K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                            ndof=ndof,solver=lin_solver,
                            springs=springs)
        # assemble forces due to body forces
        f_body = np.zeros(f.shape)
        if "density_coupled" in body_forces_kw:
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
        u[free, :], fact, precond = solve_lin(K=K, rhs=rhs[free],rhs0=u[free,:],
                                                solver=lin_solver,
                                                preconditioner=preconditioner)     
        # Objective and objective gradient
        obj = 0
        dobj[:] = 0.
        for i in np.arange(f.shape[1]):
            # obj. value, selfadjoint variables, self adjoint flag
            obj,rhs_adj,self_adj = obj_func(obj=obj, i=i,
                                            xPhys=xPhys,u=u,
                                            KE=KE, edofMat=edofMat,
                                            Kes=Kes,
                                            matinterpol=matinterpol,
                                            matinterpol_kw=matinterpol_kw,
                                            **obj_kw)
            # update sensitivity for quantities that need a small offset to
            # avoid degeneracy of the FE problem
            #"""
            # if problem not self adjoint, solve for adjoint variables and
            # calculate derivatives, else use analytical solution
            # if problem not self adjoint, solve for adjoint variables and
            # calculate derivatives, else use analytical solution
            if self_adj:
                lamU = np.zeros(f.shape)
                lamU[free,i] = rhs_adj[free,i]
            else:
                lamU = np.zeros(f.shape)
                lamU[free],_,_ = solve_lin(K, rhs=rhs_adj[free],
                                        solver=lin_solver, P=precond,
                                        preconditioner = preconditioner)
            # update sensitivity for quantities that need a small offset to
            # avoid degeneracy of the FE problem
            # standard contribution of element stiffness/conductivity
            # add explicit term
            dobj_offset = np.matvec(KE,u[edofMat,i])
            dobj[:,0] += (dscale*lamU[edofMat,i]*dobj_offset).sum(axis=1)
            # update sensitivity for quantities that do not need a small
            # offset to avoid degeneracy of the FE problem
            if "density_coupled" in body_forces_kw:
                dobj[:,0] -= simp_dx(xPhys=xPhys, eps=0., penal=1.)[:,0]*\
                                np.dot(lamU[edofMat,i],fe_dens[:,i]) 
            if debug:
                print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                        loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        
        ue = u[edofMat, 0]
        # interpolated constitutive tensor
        C_es = scale[:, :, None] * cs
        # element strain
        strain = np.einsum('eij,ej->ei', B, ue, optimize=True)
        # element stress
        stress = np.einsum('eij,ej->ei', C_es, strain, optimize=True)
        # von Mises stress and derivative
        stress_vm = von_mises_stress(stress=stress, ndim=ndim)

        if use_stress_constraint:
            i=0
            dsvm = dsvm_ds(stress=stress,stress_vm=stress_vm,ndim=ndim)
            stress_p, rhs_adj, dsP_dx,_ = stress_pnorm(
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
                obj=0,
                dscale=dscale,
                cs=cs,
                strain=strain,
                **obj_kw
            )
            lamU = np.zeros(f.shape)
            lamU[free],_,_ = solve_lin(K, rhs=rhs_adj[free],
                                    solver=lin_solver, P=precond,
                                    preconditioner = preconditioner)
            dconstr_stress_dx[:, 0] = dsP_dx[:, 0]
            dobj_offset = np.matvec(KE,u[edofMat,i])

            dconstr_stress_dx[:,0] += (dscale*lamU[edofMat,i]*dobj_offset).sum(axis=1)
            # update sensitivity for quantities that do not need a small
            # offset to avoid degeneracy of the FE problem
            if "density_coupled" in body_forces_kw:
                dconstr_stress_dx[:,0] -= simp_dx(xPhys=xPhys, eps=0., penal=1.)[:,0]*\
                                np.dot(lamU[edofMat,i],fe_dens[:,i]) 
            
            # c_stress is used to detect the max stress, details please refer to the paper
            stress_vm_max  = float(np.max(stress_vm))
            alpha_c = 0.85
            c_target = stress_vm_max / stress_p
            if c_stress is None:
                c_stress = c_target
            else:
                c_stress = alpha_c * c_stress + (1.0 - alpha_c) * c_target
            dconstr_stress_dx = c_stress * dconstr_stress_dx / stress_allow
            constr_stress = c_stress * stress_p / stress_allow - 1.0

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
        vol_up = xPhys.mean() - volfrac
        vol_lo = volfrac - xPhys.mean() - 1e-5
        constr_list.extend([vol_up, vol_lo])
        if use_stress_constraint:
            constr_list.append(float(constr_stress))
        constrs = np.asarray(constr_list, dtype=float).reshape(-1, 1)
        dconstr = np.zeros((n, n_constr), dtype=float)
        col = 0
        dconstr[:n, col] =  1.0 / n; col += 1      # vol upper
        dconstr[:n, col] = -1.0 / n; col += 1      # vol lower
        if use_stress_constraint:
            dconstr[:n, col] = dconstr_stress_dx[:, 0]
            col += 1
        #
        if ft == 1 and filter_mode == "matrix":
            dobj[:] = np.asarray(H*(dobj/Hs))
            # dconstr[:n,:] = np.asarray(H*(dconstr[:n,:]/Hs))
            dconstr[:] = np.asarray(H*(dconstr/Hs))
            # for j in range(dconstr.shape[-1]):
            #     dconstr[:nel, j] = np.asarray(H*((dconstr[:nel, j])[:, None] / Hs))[:, 0]
        elif ft == 5:
            xTilde[:] = np.asarray(H * x/ Hs)
            xBase[:] = xTilde.copy()
            dx = beta * (1 - np.tanh(beta * (xBase - eta))**2) /\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))   
            dobj[:] = dobj * dx   # dJ/d(xP_input)
            dobj[:] = np.asarray(H * (dobj / Hs))
            for j in range(dconstr.shape[-1]):
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
        # method of moving asymptotes
        if optimizer == "mma":
            nvars = n
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                                 n=nvars,
                                                                 iter=loop,
                                                                 xval=x,
                                                                 xold1=xhist[-1],
                                                                 xold2=xhist[-2],
                                                                 f0val=obj,
                                                                 df0dx=dobj,
                                                                 fval=constrs,
                                                                 dfdx=dconstr.T,
                                                                 **optimizer_kw)

            x[:] = np.asarray(xmma, dtype=float).reshape(nvars, 1) 
            xhist.pop(0); xhist.append(x.copy())
            optimizer_kw["low"] = low; optimizer_kw["upp"] = upp
            if len(xhist)> nhistory+1:
                xhist = xhist[-nhistory-1:]
            # print(f"fval: {constrs}, max_violation: {np.maximum(constrs,0).max():.3e}")    
        if debug:
            print("Post Density Update: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Filter design variables
        if ft == 1 and filter_mode == "matrix":
            xPhys[:] = np.asarray(H*x/Hs)      
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
        change = np.abs(xhist[-1][:, [0]] - xhist[-2][:, [0]]).max()
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
            im.set_array(-xPhys.reshape((nely,nelx),order="F"))
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if write_log:
            to_log("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch: {3:.10f}".format(
                         loop+1, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01 and beta is None and loop >200:
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
    Kes = KE*scale[:,:,None]
    # update physical properties of the elements and thus the entries
    # of the elements 
    sK = (scale[:,:,None] * KE).flatten() 
    # Setup and solve FE problem
    # To Do: loop over boundary conditions if incompatible
    # assemble system matrix
    K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                        ndof=ndof,solver=lin_solver,
                        springs=springs)
    u0 = None
    f_body = np.zeros(f.shape)
    if "density_coupled" in body_forces_kw:
        fes = fe_dens[None,:,:]*simp(xPhys=xThresh, eps=0., penal=1.)[:,:,None]
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
    
    dscale = matinterpol_dx(xPhys=xThresh, **matinterpol_kw)

    obj = 0.
    obj,rhs_adj,self_adj = obj_func(obj=obj, i=0,
                                    xPhys=xThresh,u=u_bw,
                                    KE=KE, edofMat=edofMat,
                                    Kes=Kes,
                                    matinterpol=matinterpol,
                                    matinterpol_kw=matinterpol_kw,
                                    **obj_kw)
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
    return obj

# The real main driver
if __name__ == "__main__":
    #
    #sketch(save=True)
    # Default input parameters
    nelx=100
    nely=100
    nelz=None
    volfrac=0.3
    rmin=5.0
    penal=3.
    ft=1 
    elem_size=1.0
    nouteriter=2000
    export=True
    write_log=True
    display=True
    use_stress_constraint = True
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
    el_flags = np.zeros(nelx*nely, dtype=np.int32)
    xs = np.arange(nelx//5*2+1, nelx, dtype=int)   
    ys = np.arange(0, nely//5*3, dtype=int)   
    idx = (xs[:, None] * nely+ ys[None, :]).astype(int).ravel()
    el_flags[idx] = 1

    obj = main(nelx=nelx,nely=nely,nelz=nelz,volfrac=volfrac,penal=penal,rmin=rmin,ft=ft,
         obj_func=compliance,
         body_forces_kw={"density_coupled": np.array([0,-1e-7])},
         el_flags = el_flags,
         display=display,
         bcs=bcs,
         file='Lbracket',
         nouteriter=nouteriter,
         use_stress_constraint=use_stress_constraint,
         export=export,write_log=write_log)
    # for tests
    np.savetxt("stress_lbracket_obj.csv", np.array([obj]), delimiter=",")
    

