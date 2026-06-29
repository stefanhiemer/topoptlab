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
# different elements/physics
from topoptlab.stiffness_tensors import orthotropic_2d,orthotropic_3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_aniso_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_aniso_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d

from topoptlab.material_interpolation import simp,simp_dx,ramp,ramp_dx
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# optimizers
from mmapy import mmasub
from topoptlab.objectives import stress_pnorm,compliance,var_squarederror
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk,threshold
# map element data to img/voxel
from topoptlab.utils import check_optimizer_kw
# logging related stuff
from topoptlab.log_utils import EmptyLogger,init_logging
# rotation matrices and derivatives
from topoptlab.rotation_matrices import R_2d, dR2_dtheta, Rv_2d, dRvdtheta_2d,R_3d,dR3_dtheta,dR3_dphi,Rv_3d,dRvdtheta_3d,dRvdphi_3d


# MAIN DRIVER
def main(nelx: int, nely: int, nelz: int | None,
        volfrac: float, rmin: float, ft: int,
        Emax: float = 1.0, nu: float = 0.3, Eratio: float = 0.05,
        filter_mode: str = "matrix", lin_solver: str = "cvxopt-cholmod",
        preconditioner: str | None = None,
        assembly_mode: str = "full", body_forces_kw: Dict | None = None,
        bcs: Callable = Lbracket, l: float | np.ndarray = 1.0,
        obj_func: Callable = stress_pnorm, obj_kw={},         
        el_flags: np.ndarray | None = None,
        optimizer: str = "mma", optimizer_kw: Dict | None = None,
        nouteriter: int = 2000,
        file: str = "lbracket",
        matinterpol: Callable = simp, matinterpol_dx: Callable = simp_dx,
        matinterpol_kw: Dict = {"eps": 1e-9, "penal": 3.},
        display: bool = False, export: bool = False,
        write_log: bool = False,
        debug: int = 0) -> float:
    """
    Run topology optimization with compliance minimization 
    with control of orientations of stiffness.
    Use el_flags to fix the orientation to compare if improves the obj.

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
    Eratio : float, optional
        a ratio to scale the modulus for the weak direction.
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
    nouteriter : int, optional
        Maximum number of outer topology optimization iterations.
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
        n = nelx * nely
        create_edofMat = create_edofMat2d
        xe = np.array([[[-1.,-1.],
                        [1.,-1.],
                        [1.,1.],
                        [-1.,1.]]])/2
        from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
    else:
        ndim = 3
        n = nelx * nely * nelz
        create_edofMat = create_edofMat3d 
        xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])/2  
        from topoptlab.elements.trilinear_hexahedral import shape_functions_dxi 

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
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])

    xe = l*xe*np.ones((n,1,1))
    # Allocate design variables (as array, 2nd column as rotation field), initialize and allocate sens.
    x = volfrac * np.ones((n,2), dtype=float,order='F')
    xold, xPhys, xBase,xTilde = x.copy(), x.copy(), x.copy(), x.copy()
    x[:, 1] = 0.
    xi,etaa,zeta = ndim*[np.array([0.])] + int(3-ndim)*[None]
    B = infini_strain_matrix(xi=xi, 
                             eta=etaa, 
                             zeta=zeta, 
                             xe=xe, 
                             all_elems=True, 
                             shape_functions_dxi=shape_functions_dxi)
    if ft == 5:
        beta = 1
        eta = find_eta(eta0=0.5, xTilde=xTilde[:, [0]], beta=beta, volfrac=volfrac)
    else:
        beta = None
    #
    if ndim ==2:
        # stiffness tensor
        cs = [orthotropic_2d(Ex=Emax, Ey=Eratio*Emax, nu_xy=nu, G_xy=0.05) \
              for i in np.arange(int(nely))]
        cs = np.tile(np.stack(cs),(nelx,1,1))

    if ndim ==3:
        # stiffness tensor
        cs = [orthotropic_3d(Ex=Emax, Ey=Eratio*Emax, Ez=Eratio*Emax,
                             nu_xy=nu, nu_xz=nu, nu_yz=nu,
                             G_xy=0.05, G_xz=0.05, G_yz=0.05) \
              for i in np.arange(int(nely))]
        cs = np.tile(np.stack(cs),(nelx*nelz,1,1))
    # initialize arrays for gradients
    dobj = np.zeros_like(x)
    # initialize solver
    # upper, lower volume constr 
    n_constr = 2
    nvars =x .shape[0] * x.shape[1]
    optimizer_kw = check_optimizer_kw(optimizer=optimizer,
                                      n=nvars,
                                      ft=ft,
                                      n_constr=n_constr,
                                      optimizer_kw=optimizer_kw)

    if optimizer in ["mma"]:
        # mma needs results of the two previous iterations
        max_history = 3
        xhist = [x.copy() for i in np.arange(max_history)] 
        optimizer_kw["xmin"][n:] = 0.0
        optimizer_kw["xmax"][n:] = 1.0
        optimizer_kw["move"] = 0.05
        # handle element element flags
        if el_flags is not None:
            # passive, for n:, fix the theta to 0
            mask = el_flags == 1
            optimizer_kw["xmin"][mask] = 0.
            optimizer_kw["xmax"][mask] = 0.+1e-9

            # active
            mask = el_flags == 2
            optimizer_kw["xmin"][mask] = 1.- 1e-9
            optimizer_kw["xmax"][mask] = 1.
        
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
    ndof = np.prod(np.array([nelx,nely,nelz])[:ndim]+1)*n_ndof
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,
                                             nely=nely,
                                             nelz=nelz,
                                             nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat,
                              mode="full")
    
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
        #
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                      rmin=rmin,ndim=ndim, 
                                      kernel_fn=hat_kernel)
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx,nely=nely,nelz=nelz,
                                 ndof=ndof)
    if display:
            # Initialize plot and plot the initial design
            plt.ion()  # Ensure that redrawing is possible
            fig,ax = plt.subplots()
            im = ax.imshow(-xPhys[:,[0]].reshape((nely,nelx),order="F"), cmap='gray',
                        interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
            ax.tick_params(axis='both',
                        which='both',
                        bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False)
            fig.show()

    # optimization loop
    loopbeta = 0
    for loop in np.arange(nouteriter):
        #
        loopbeta += 1  
        if ndim == 2:
            theta = np.pi * xPhys[:,[1]]
            R = R_2d(theta)
            dRdtheta = dR2_dtheta(theta)
            Rv = Rv_2d(theta, eng_conv=False)
            Rv_e = Rv_2d(theta, eng_conv=True)
            dRvdtheta = dRvdtheta_2d(theta, eng_conv=False)
            dRvdtheta_e = dRvdtheta_2d(theta, eng_conv=True)
        else:
            # in 3D, use phi only
            phi = np.pi * xPhys[:,[1]]
            theta = np.zeros_like(phi)
            R = R_3d(theta,phi)
            dRdtheta = dR3_dtheta(theta,phi)
            Rv   = Rv_3d(theta, phi, eng_conv=False)   # stress-like
            Rv_e = Rv_3d(theta, phi, eng_conv=True)    # strain-like
            dRvdtheta   = dRvdtheta_3d(theta, phi, eng_conv=False)
            dRvdtheta_e = dRvdtheta_3d(theta, phi, eng_conv=True)
            dRvdphi   = dRvdphi_3d(theta, phi, eng_conv=False)
            dRvdphi_e = dRvdphi_3d(theta, phi, eng_conv=True)

        tmp = np.einsum('eij,ejk->eik', Rv, cs, optimize=True)
        cs_rot = np.einsum('eij,ejk->eik', tmp, np.linalg.inv(Rv_e), optimize=True)

        # element matrices with current orientations
        if ndim == 2:
            KE  = np.empty((n, 8, 8), dtype=float)
            for e in np.arange(n):
                KE[e, :, :] = lk_linear_elast_aniso_2d(c=cs_rot[e, :, :], l=l,g=np.array([0.]),t=1.0)       
        else:
            KE  = np.empty((n, 24, 24), dtype=float)
            for e in np.arange(n):
                KE[e, :, :] = lk_linear_elast_aniso_3d(c=cs_rot[e, :, :], l=l,g=np.array([0.,0.]))

        # calculate / interpolate material properties
        dscale = matinterpol_dx(xPhys=xPhys[:,[0]], **matinterpol_kw)
        scale = matinterpol(xPhys=xPhys[:,[0]],**matinterpol_kw)
        Kes = KE*scale[:,:,None]

        # solve FEM, calculate obj. func. and gradients.
        # for
        # update physical properties of the elements and thus the entries
        # of the elements
        if assembly_mode == "full":
            # this here is more memory efficient than Kes.flatten() as it
            # provides a view onto the original Kes array instead of a copy
            sK = Kes.reshape(np.prod(Kes.shape))
        # Setup and solve FE problem
        # assemble system matrix
        K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                            ndof=ndof,solver=lin_solver,
                            springs=springs)
        # assemble forces due to body forces
        f_body = np.zeros(f.shape)
        if "density_coupled" in body_forces_kw:
            fes = fe_dens[None,:,:]*simp(xPhys=xPhys[:,[0]], eps=0., penal=1.)[:,:,None]
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
            # obj. value, selfadjoint variables, self adjoint flag
            obj,rhs_adj,self_adj = obj_func(obj=obj, i=i,
                                            xPhys=xPhys[:,[0]],u=u,
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
            # contribution due to force induced by strain
            dobj[:,0] += (dscale*lamU[edofMat,i]*dobj_offset).sum(axis=1)
            # update sensitivity for quantities that do not need a small
            # offset to avoid degeneracy of the FE problem
            if "density_coupled" in body_forces_kw:
                dobj[:,0] -= simp_dx(xPhys=xPhys[:,[0]], eps=0., penal=1.)[:,0]*\
                                np.dot(lamU[edofMat,i],fe_dens[:,i]) 
            if debug:
                print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                        loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))

        ue = u[edofMat, 0]
        # interpolated constitutive tensor
        C_es = scale[:, :, None] * cs_rot
        # element strain
        strain = np.einsum('eij,ej->ei', B, ue, optimize=True)
        # element stress
        stress = np.einsum('eij,ej->ei', C_es, strain, optimize=True)
        # von Mises stress and derivative
        stress_vm = von_mises_stress(stress=stress, ndim=ndim)

        if ndim == 2:
            # dC/dθ = dRv*cs*Rv^T + Rv*cs*(dRv)^T
            dRv_e_inv = -np.einsum('eij,ejk,ekl->eil', np.linalg.inv(Rv_e), dRvdtheta_e, np.linalg.inv(Rv_e), optimize=True)
            term1 = np.einsum('eij,ejk->eik', dRvdtheta, cs, optimize=True)
            term1 = np.einsum('eij,ejk->eik', term1, np.linalg.inv(Rv_e), optimize=True)
            term2 = np.einsum('eij,ejk->eik', Rv, cs, optimize=True)
            term2 = np.einsum('eij,ejk->eik', term2, dRv_e_inv, optimize=True)
            dC_dtheta = term1 + term2
        else:
            # in 3D, use phi only
            dRv_e_inv_theta = -np.einsum(
                'eij,ejk,ekl->eil',
                np.linalg.inv(Rv_e), dRvdtheta_e, np.linalg.inv(Rv_e),
                optimize=True
            )
            dRv_e_inv_phi = -np.einsum(
                'eij,ejk,ekl->eil',
                np.linalg.inv(Rv_e), dRvdphi_e, np.linalg.inv(Rv_e),
                optimize=True
            )
            # dC/dtheta
            # term1 = np.einsum('eij,ejk->eik', dRvdtheta, cs, optimize=True)
            # term1 = np.einsum('eij,ejk->eik', term1, np.linalg.inv(Rv_e), optimize=True)

            # term2 = np.einsum('eij,ejk->eik', Rv, cs, optimize=True)
            # term2 = np.einsum('eij,ejk->eik', term2, dRv_e_inv_theta, optimize=True)

            # dC_dtheta = term1 + term2
            # dC/dphi
            term1 = np.einsum('eij,ejk->eik', dRvdphi, cs, optimize=True)
            term1 = np.einsum('eij,ejk->eik', term1, np.linalg.inv(Rv_e), optimize=True)

            term2 = np.einsum('eij,ejk->eik', Rv, cs, optimize=True)
            term2 = np.einsum('eij,ejk->eik', term2, dRv_e_inv_phi, optimize=True)
            dC_dphi = term1 + term2

        if ndim == 2:
            # dKe/dθ
            dKe_dtheta  = np.empty((n, 8, 8), dtype=float)
            for e in np.arange(n):
                dKe_dtheta[e, :, :] = lk_linear_elast_aniso_2d( c=dC_dtheta[e, :, :], l=l,g=np.array([0.]),t=1.0)
            dKe_dtheta *= scale[:,:,None]  
        else:  # 3D
            # dKe/dφ, dKe/dθ
            dKe_dphi  = np.empty((n, 24, 24), dtype=float)
            # dKe_dtheta  = np.empty((n, 24, 24), dtype=float)
            for e in np.arange(n):
                dKe_dphi[e, :, :] = lk_linear_elast_aniso_3d(c=dC_dphi[e, :, :], l=l,g=np.array([0.,0.]))
                # dKe_dtheta[e, :, :] = lk_linear_elast_aniso_3d(c=dC_dtheta[e, :, :], l=l,g=np.array([0.,0.]))
            dKe_dphi   *= scale[:,:,None]
            # dKe_dtheta *= scale[:,:,None]

        # angle gradient of objective
        lamU_e = lamU[edofMat,0] 
        if ndim == 2:
            term_K = np.einsum('ei,eij,ej->e', lamU_e, dKe_dtheta, ue, optimize=True)
            dobj[:, [1]] = np.pi * term_K[:, None]
        else:
            term_K = np.einsum('ei,eij,ej->e', lamU_e, dKe_dphi, ue, optimize=True)
            dobj[:, [1]] = np.pi * term_K[:, None]

        if loop == 0:
            if export:
                if ndim == 2:
                    ct, st = np.cos(theta), np.sin(theta)
                    dir_vec = np.concatenate((ct, st, np.zeros_like(ct)), axis=1)
                else:
                    ct, st = np.cos(theta), np.sin(theta)
                    cp, sp = np.cos(phi), np.sin(phi)
                    dir_vec = np.concatenate((ct*cp, st*cp, -sp), axis=1)
                export_vtk(
                    filename=f"{file}_it{loop+1:04d}",
                    nelx=nelx, nely=nely, nelz=nelz,
                    xPhys=xPhys[:,[0]], x=x[:,[0]],
                    vectors=dir_vec,
                    elem_size=l,
                    stress_vm=stress_vm,
                    u=u, f=f+f_body,
                    volfrac=volfrac)

        # Build constraint values 
        constr_list = []
        vol_up = xPhys[:,[0]].mean() - volfrac - 1e-5
        vol_lo = volfrac - xPhys[:,[0]].mean() - 1e-5
        constr_list.extend([vol_up,vol_lo])

        constrs = np.asarray(constr_list, dtype=float).reshape(-1, 1)

        dconstr = np.zeros((nvars, n_constr), dtype=float)
        col = 0
        dconstr[:n, col] =  1.0 / n; col += 1      # vol upper
        dconstr[:n, col] = -1.0 / n; col += 1      # vol lower


        if ft == 1 and filter_mode == "matrix":
            dobj[:] = np.asarray(H*(dobj/Hs))
            dconstr[:n,:] = np.asarray(H*(dconstr[:n,:]/Hs))
        elif ft == 5:
            xTilde[:] = np.asarray(H * x/ Hs)
            xBase[:] = xTilde.copy()
            dx = beta * (1 - np.tanh(beta * (xBase[:, [0]] - eta))**2) /\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))   
            dobj[:, [0]] = dobj[:, [0]] * dx
            dobj[:, [0]] = np.asarray(H * (dobj[:, [0]] / Hs))
            dobj[:, [1]] = np.asarray(H * (dobj[:, [1]] / Hs))
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
        # method of moving asymptotes
        if optimizer == "mma":
            xval  = x.reshape((nvars, 1), order='F')
            xold1 = xhist[-1].reshape((nvars, 1), order='F')
            xold2 = xhist[-2].reshape((nvars, 1), order='F')
            df0dx = dobj.reshape((nvars, 1), order='F')
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                                 n=nvars,
                                                                 iter=loop,
                                                                 xval=xval,
                                                                 xold1=xold1,
                                                                 xold2=xold2,
                                                                 f0val=obj,
                                                                 df0dx=df0dx,
                                                                 fval=constrs,
                                                                 dfdx=dconstr.T,
                                                                 **optimizer_kw)

            # update asymptotes
            optimizer_kw["low"] = low
            optimizer_kw["upp"] = upp
            x = xmma.reshape((n, 2), order='F')
            xhist.append(x.copy())
            # prune history if too long
            if len(xhist)> max_history+1:
                xhist = xhist[-max_history-1:]
            #
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
            eta = find_eta(eta0=eta, xTilde=xBase[:, [0]], beta=beta, volfrac = volfrac)
            xPhys[:, [0]] = (np.tanh(beta*eta)+np.tanh(beta * (xBase[:, [0]] - eta)))/\
                       (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
            xPhys[:, [1]] = xBase[:, [1]]
        elif ft == -1:
            xPhys[:]  = x
        if debug:
            print("Post Density Filter: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
            
        # Compute the change by the inf. norm
        change = np.abs(xhist[-1][:, [0]] - xhist[-2][:, [0]]).max()
        change_ang = np.abs(xhist[-1][:, [1]] - xhist[-2][:, [1]]).max()
        export_every = 100
        if export and ((loop + 1) % export_every == 0):
            if ndim == 2:
                ct, st = np.cos(theta), np.sin(theta)
                dir_vec = np.concatenate((ct, st, np.zeros_like(ct)), axis=1)
            else:
                ct, st = np.cos(theta), np.sin(theta)
                cp, sp = np.cos(phi), np.sin(phi)
                dir_vec = np.concatenate((ct*cp, st*cp, -sp), axis=1)
            export_vtk(
                filename=f"{file}_it{loop+1:04d}",
                nelx=nelx, nely=nely, nelz=nelz,
                xPhys=xPhys[:,[0]], x=x[:,[0]],
                elem_size=l,
                vectors=dir_vec,
                stress_vm=stress_vm,
                u=u, f=f+f_body,
                volfrac=volfrac)


        # Plot to screen
        if display:
            im.set_array(-xPhys[:,[0]].reshape((nely,nelx),order="F"))
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if write_log:
            to_log("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch_x: {3:.10f} ch_ang: {4:.10f}".format(
                         loop+1, obj, xPhys[:, [0]].mean(), change, change_ang))
        # convergence check
        if change < 0.01 and beta is None and loop >200 and change_ang < 0.01:
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
    xThresh = threshold(xPhys[:,[0]],volfrac)
    scale = matinterpol(xPhys=xThresh[:,[0]],**matinterpol_kw)
    # update physical properties of the elements and thus the entries
    # of the elements

    if ndim == 2:
        theta_final = np.pi * xPhys[:, [1]]
        Rv_final = Rv_2d(theta_final, eng_conv=False)
        Rv_e_final = Rv_2d(theta_final, eng_conv=True)
        Rv_e_inv_final = np.linalg.inv(Rv_e_final)

        tmp = np.einsum('eij,ejk->eik', Rv_final, cs, optimize=True)
        cs_rot_final = np.einsum('eij,ejk->eik', tmp, Rv_e_inv_final, optimize=True)

        KE_final = np.empty((n, 8, 8), dtype=float)
        for e in np.arange(n):
            KE_final[e, :, :] = lk_linear_elast_aniso_2d(c=cs_rot_final[e, :, :], l=l, g=np.array([0.]), t=1.0)
    else:
        phi_final = np.pi * xPhys[:, [1]]
        theta_final = np.zeros_like(phi_final)
        Rv_final   = Rv_3d(theta_final, phi_final, eng_conv=False)
        Rv_e_final = Rv_3d(theta_final, phi_final, eng_conv=True)
        Rv_e_inv_final = np.linalg.inv(Rv_e_final)

        tmp = np.einsum('eij,ejk->eik', Rv_final, cs, optimize=True)
        cs_rot_final = np.einsum('eij,ejk->eik', tmp, Rv_e_inv_final, optimize=True)

        KE_final = np.empty((n, 24, 24), dtype=float)
        for e in np.arange(n):
            KE_final[e, :, :] = lk_linear_elast_aniso_3d(
                c=cs_rot_final[e, :, :], l=l, g=np.array([0., 0.])
            )
    
    if assembly_mode == "full":
        sK = (scale[:,:,None] * KE_final).flatten() 
    # Setup and solve FE problem
    # To Do: loop over boundary conditions if incompatible
    # assemble system matrix
    K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                        ndof=ndof,solver=lin_solver,
                        springs=springs)
    u0 = None            
    f_body = np.zeros(f.shape)
    if "density_coupled" in body_forces_kw:
        fes = fe_dens[None,:,:]*simp(xPhys=xThresh[:,[0]], eps=0., penal=1.)[:,:,None]
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
    C_es = scale[:, :, None] * cs_rot_final
    ue = u_bw[edofMat, i]
    # element strain
    strain = np.einsum('eij,ej->ei', B, ue, optimize=True)
    # element stress
    stress = np.einsum('eij,ej->ei', C_es, strain, optimize=True)
    # von Mises stress
    stress_vm = von_mises_stress(stress=stress, ndim=ndim)
    obj = 0.
    Kes_final = KE_final * scale[:, :, None]
    obj,rhs_adj,self_adj = obj_func(obj=obj, i=0,
                                    xPhys=xThresh,u=u_bw,
                                    KE=KE_final, edofMat=edofMat,
                                    Kes=Kes_final,
                                    matinterpol=matinterpol,
                                    matinterpol_kw=matinterpol_kw,
                                    **obj_kw)
        #
    if write_log:
        to_log("final.: obj.: {0:.10f} vol.: {1:.10f}".format(obj, xThresh.mean()))
    #
    if export:
        if ndim == 2:
            ct, st = np.cos(theta_final), np.sin(theta_final)
            dir_vec = np.concatenate((ct, st, np.zeros_like(ct)), axis=1)
        else:
            ct, st = np.cos(theta_final), np.sin(theta_final)
            cp, sp = np.cos(phi_final), np.sin(phi_final)
            dir_vec = np.concatenate((ct*cp, st*cp, -sp), axis=1)
        export_vtk(
            filename=f"{file}_it{loop+1:04d}",
            nelx=nelx, nely=nely, nelz=nelz,
            xPhys=xPhys[:, [0]], x=x[:, [0]],
            elem_size=l,
            stress_vm=stress_vm,
            vectors=dir_vec,
            u=u_bw, f=f+f_body,
            volfrac=volfrac) 
    return obj

# The real main driver
if __name__ == "__main__":
    #
    #sketch(save=True)
    # Default input parameters
    nelx = 60
    nely = int(nelx/3)
    nelz=None
    volfrac=0.5
    rmin=2.4
    ft=1 # ft==0 -> sens, ft==1 -> dens
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
    if len(sys.argv)>5: ft     =int(sys.argv[5])
    if len(sys.argv)>6: nouteriter =int(sys.argv[6])
    if len(sys.argv) > 7: export = bool(int(sys.argv[7]))
    if len(sys.argv) > 8: write_log = bool(int(sys.argv[8]))
    if len(sys.argv) > 9: display = bool(int(sys.argv[9]))
    #
    if nelz is None:
        bcs=mbb_2d
    else:
        raise ValueError("Only for 2D validation")
    
    el_flags = np.zeros(nelx*nely*2, dtype=int)
    el_flags[nelx*nely:,] = 1   # fix all angle vars to 0

    obj = main(nelx=nelx,nely=nely,nelz=nelz,volfrac=volfrac,rmin=rmin,ft=ft,
         obj_func=compliance,
         body_forces_kw={"density_coupled": np.array([0,-1e-7])},
         display=display,
         el_flags=el_flags,
         bcs=bcs,
         file='mbb_aniso',
         nouteriter=nouteriter,
         export=export,write_log=write_log)
    # for tests
    np.savetxt("stiffness_rotation_obj.csv", np.array([obj]), delimiter=",")
    

