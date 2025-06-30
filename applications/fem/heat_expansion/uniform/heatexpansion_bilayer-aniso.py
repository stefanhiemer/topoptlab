import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import selffolding_2d, selffolding_3d
# different elements/physics
from topoptlab.stiffness_tensors import orthotropic_2d, orthotropic_3d
from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d,_lf_strain_2d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d,_lf_strain_3d
from topoptlab.elements.bodyforce_2d import lf_bodyforce_2d
from topoptlab.elements.bodyforce_3d import lf_bodyforce_3d
from topoptlab.elements.heatexpansion_2d import _fk_heatexp_2d
from topoptlab.elements.heatexpansion_3d import _fk_heatexp_3d
from topoptlab.material_interpolation import simp,simp_dx,ramp,ramp_dx
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_heat_expansion(nelx, nely, nelz=None,
                       xPhys=None, penal=3., eps=1e-9,
                       Emax=1.0, Emin=1e-9, nu=0.3, G=0.05,
                       a1=2.5e-2, a2=5e-2,
                       Eratio = 0.05,
                       lin_solver="scipy-direct", preconditioner=None,
                       assembly_mode="full",
                       body_forces_kw={},
                       bc=selffolding_2d, l=1.,
                       file="fem_heat-expansion_bilayer-aniso",
                       export=True):
    """
    Run a single finite element simulation on a regular grid.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction. Only important if ndim is 3.
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    penal : float
        penalty exponent for the SIMP method.
    Emax : float
        (maximum) Young's modulus. If xPhys is None, all elements take this
        value.
    Emin : float
        minimum Young's modulus for the modified SIMP approach.
    nu : float
        Poissson's ratio.
    a1 : float
        heat expansion coefficient of phase 1
    a2 : float
        heat expansion coefficient of phase 2
    Eratio : float
        ratio of Young's moduli from 1:2. So 0.35 means the Young's modulus of
        phase 2 is 0.35 and the one of phase 1 is 1.
    solver : str
        solver for linear systems. Check function lin solve for available
        options.
    preconditioner : str or None
        preconditioner for linear systems.
    assembly_mode : str
        whether full or only lower triangle of linear system / matrix is
        created.
    bc : str or callable
        returns the boundary conditions
    file : str
        name of output files
    export : bool
        if True, export design as vtk file.

    Returns
    -------
    None.

    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
    elif ndim == 3:
        n = nelx * nely * nelz
    #
    if ndim == 2:
        xe = l*np.array([[[-1.,-1.],
                        [1.,-1.],
                        [1.,1.],
                        [-1.,1.]]])/2 * np.ones(n)[:,None,None]
    elif ndim == 3:
        xe = l*np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])/2 \
            * np.ones(n)[:,None,None]
    #
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    #
    if xPhys is None:
        xPhys = np.ones(n, dtype=float,order='F')
    # isotropic bilayer
    if ndim ==2:
        # stiffness tensor
        cs = [orthotropic_2d(Ex=1., Ey=Eratio, nu_xy=nu, G_xy=G) \
              for i in np.arange(int(nely/2))]
        cs += [orthotropic_2d(Ex=Eratio, Ey=1., nu_xy=nu*Eratio, G_xy=G) \
               for j in np.arange(int(nely/2),nely)]
        cs = np.tile(np.stack(cs),(nelx,1,1))
        # lin. expansion coefficient
        a = np.stack([np.diag([a1,a2]) for i in np.arange(int(nely/2))]+\
                     [np.diag([a2,a1]) for i in np.arange(int(nely/2))])
        a = np.tile(a,(nelx,1,1))
    if ndim ==3:
        # stiffness tensor
        cs = [orthotropic_3d(Ex=1.0, Ey=Eratio, Ez=Eratio,
                             nu_xy=nu, nu_xz=nu, nu_yz=nu,
                             G_xy=G, G_xz=G, G_yz=G) \
              for i in np.arange(int(nely/2))]
        cs += [orthotropic_3d(Ex=Eratio, Ey=1.0, Ez=Eratio,
                              nu_xy=nu*Eratio, nu_xz=nu, nu_yz=nu,
                              G_xy=G, G_xz=G, G_yz=G) \
               for j in np.arange(int(nely/2),nely)]
        cs = np.tile(np.stack(cs),(nelx*nelz,1,1))
        # lin. expansion coefficient
        a = np.stack([np.diag([a1,a2,a2]) for i in np.arange(int(nely/2))]+\
                     [np.diag([a2,a1,a2]) for i in np.arange(int(nely/2))])
        a = np.tile(a,(nelx*nelz,1,1))
    # get element stiffness matrix and element of freedom matrix
    nT_ndof = 1
    if ndim == 2:
        KE = _lk_linear_elast_2d(xe=xe,c=cs)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nE_ndof = int(KE.shape[-1]/4)
        # number of degrees of freedom
        nEdof = (nelx+1)*(nely+1)*nE_ndof
        nTdof = (nelx+1)*(nely+1)*nT_ndof
        # element degree of freedom matrix plus some helper indices
        EedofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                    nnode_dof=nE_ndof)
        TedofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                    nnode_dof=1)
    elif ndim == 3:
        KE = _lk_linear_elast_3d(xe=xe,c=cs)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nE_ndof = int(KE.shape[-1]/8)
        # number of degrees of freedom
        nEdof = (nelx+1)*(nely+1)*(nelz+1)*nE_ndof
        nTdof = (nelx+1)*(nely+1)*(nelz+1)*nT_ndof
        # element degree of freedom matrix plus some helper indices
        EedofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                    nelz=nelz,
                                                    nnode_dof=nE_ndof)
        TedofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                    nelz=nelz,
                                                    nnode_dof=1)
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
            if ndim == 2 and nE_ndof != 1:
                fixed = np.array([0,1,3])
            elif ndim == 3 and nE_ndof != 1:
                fixed = np.array([0,1,2,4,5,7,8])
            elif nE_ndof == 1:
                fixed = np.array([0])
            free = np.setdiff1d(np.arange(KE.shape[-1]), fixed)
            u0 = np.zeros(fe_strain.shape)
            u0[free] = np.linalg.solve(KE[free,:][:,free],
                                       fe_strain[free,:])
        else:
            fe_strain = None
        #
        if "density_coupled" in body_forces_kw.keys():
            # fetch functions to create body force
            if ndim == 2 and nE_ndof!=1:
                lf = lf_bodyforce_2d
            elif ndim == 3 and nE_ndof!=1:
                lf = lf_bodyforce_3d
            fe_dens = lf_bodyforce_2d(b=body_forces_kw["density_coupled"],l=l)
        else:
            fe_dens = None
        #
        if len([key for key in body_forces_kw.keys() \
                if key not in ["density_coupled","strain_uniform"]]):
            raise NotImplementedError("One type of bodyforce/source has not yet been implemented.")
    #
    if assembly_mode == "symmetry":
        assm_indcs = np.column_stack(np.triu_indices_from(KE[0]))
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=EedofMat,mode=assembly_mode)
    # BC's and support
    u,f,fixedE,freeE,springs = bc(nelx=nelx,nely=nely,nelz=nelz,ndof=nEdof)
    #
    T = np.ones((nTdof,1))
    # interpolate material properties
    scale = (Emin+xPhys**penal*(Emax-Emin))
    E = Emax * scale
    # entries of stiffness matrix
    if assembly_mode == "full":
        sK = (E[:,None,None] * KE).reshape(np.prod(E.shape+KE.shape[1:]))
    elif assembly_mode == "symmetry":
        sK = (E[:,None,None] * KE)[:,assm_indcs[0],assm_indcs[1]].\
              reshape( n*nE_ndof*(nE_ndof+1) )
    #
    KE = assemble_matrix(sK=sK,iK=iK,jK=jK,
                         ndof=nEdof,solver=lin_solver,
                         springs=springs)
    # assemble right hand side
    # forces due to heat expansion per element
    if ndim == 2:
        fTe = _fk_heatexp_2d(xe=xe,
                             c=cs,
                             a=a,
                             DeltaT=T[TedofMat][:,:,0])
    elif ndim == 3:
        fTe = _fk_heatexp_3d(xe=xe,
                             c=cs,
                             a=a,
                             DeltaT=T[TedofMat][:,:,0])
    # assemble
    fT = np.zeros(f.shape)
    np.add.at(fT[:,0],
              EedofMat.flatten(),
              fTe.flatten())
    # assemble forces due to body forces
    f_body = np.zeros(f.shape)
    u0 = None
    for bodyforce in body_forces_kw.keys():
        # assume each strain is a column vector in Voigt notation
        if "strain_uniform" in body_forces_kw.keys():
            fes = fe_strain[None,:,:]*scale[:,None,None]
            np.add.at(f_body,
                      EedofMat,
                      fes)
        if "density_coupled" in body_forces_kw.keys():
            fes = fe_dens[None,:,:]*simp(xPhys=xPhys, eps=eps, penal=penal)[:,None,None]
            np.add.at(f_body,
                      EedofMat,
                      fes)
    # assemble completely
    rhsE = assemble_rhs(f0=f+fT+f_body,
                        solver=lin_solver)
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE,solver=lin_solver,
                 free=freeE,fixed=fixedE)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[freeE, :], fact, precond, = solve_lin(K=KE, rhs=rhsE[freeE],
                                            solver=lin_solver,
                                            preconditioner=preconditioner)
    print(u[2 *nelx*(nely+1) + 1,0])
    #np.savetxt("surface-displacements.csv",
    #           u[np.arange(0,2*(nelx+1)*(nely+1),2*(nely+1))+1,0])

    if export:
        export_vtk(filename=file+"T"+str(ndim),
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=T)
        export_vtk(filename=file+"E-"+str(ndim),
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=1/a[:,0,0],
                   u=u,f=f+fT)
    return

if __name__ == "__main__":
    nelx=240
    nely=int(nelx/6)
    nelz=None
    L = 60
    l = 60/nelx
    #
    import sys
    if len(sys.argv)>1:
        nelx = int(sys.argv[1])
    if len(sys.argv)>2:
        nely = int(sys.argv[2])
    if len(sys.argv)>3:
        nelz = int(sys.argv[2])
    #
    fem_heat_expansion(nelx=nelx,nely=nely,nelz=nelz,l=l,
                       body_forces_kw={"density_coupled": np.array([0,-1e-7])}
                       )
