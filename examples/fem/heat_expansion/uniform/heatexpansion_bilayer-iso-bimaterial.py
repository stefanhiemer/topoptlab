# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import selffolding_2d, selffolding_3d
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d, isotropic_3d
from topoptlab.elements.linear_elasticity_2d import _lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import _lk_linear_elast_3d
from topoptlab.elements.heatexpansion_2d import _fk_heatexp_2d
from topoptlab.elements.heatexpansion_3d import _fk_heatexp_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_heat_expansion(nelx, nely, nelz=None,
                       xPhys=None, penal=3.,
                       Emax=1.0, Emin=1e-9, nu=0.3,
                       a1=2.5e-2, a2=5e-2,
                       Eratio = 0.05,
                       lin_solver="cvxopt-cholmod", preconditioner=None,
                       assembly_mode="full",
                       bc=selffolding_3d,
                       file="fem_heat-expansion_bilayer-iso",
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
    if xPhys is None:
        xPhys = np.ones(n, dtype=float,order='F')
    #
    if ndim == 2:
        xe = np.array([[[-1.,-1.],
                        [1.,-1.],
                        [1.,1.],
                        [-1.,1.]]]) * np.ones(xPhys.shape)[:,None,None]
    elif ndim == 3:
        xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]]) \
            * np.ones(xPhys.shape)[:,None,None]
    # isotropic bilayer
    if ndim ==2:
        # stiffness tensor
        cs = [isotropic_2d(E=1., nu = nu) for i in np.arange(int(nely/2))]
        cs += [isotropic_2d(E=Eratio, nu=nu) for j in np.arange(int(nely/2),nely)]
        cs = np.tile(np.stack(cs),(nelx,1,1))
        # lin. expansion coefficient
        a = np.hstack( (np.full( (int(nely/2)), fill_value=a1 ),
                        np.full( (int(nely/2)), fill_value=a2 )))
        a = np.tile(a,(nelx))
    if ndim ==3:
        # stiffness tensor
        cs = [isotropic_3d(E=1., nu = nu) for i in np.arange(int(nely/2))]
        cs += [isotropic_3d(E=Eratio, nu=nu) for j in np.arange(int(nely/2),nely)]
        cs = np.tile(np.stack(cs),(nelx*nelz,1,1))
        # lin. expansion coefficient
        a = np.hstack( (np.full( (int(nely/2)), fill_value=a1 ),
                        np.full( (int(nely/2)), fill_value=a2 )))
        a = np.tile(a,(nelx*nelz))
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
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(EedofMat,mode=assembly_mode)
    # BC's and support
    u,f,fixedE,freeE,_ = bc(nelx=nelx,nely=nely,nelz=nelz,ndof=nEdof)
    #
    T = np.ones((nTdof,1))
    # interpolate material properties
    E = (Emin+(xPhys)** penal*(Emax-Emin))
    # entries of stiffness matrix
    if assembly_mode == "full":
        sK = (E[:,None,None] * KE).flatten()
    #
    KE = assemble_matrix(sK=sK,iK=iK,jK=jK,
                         ndof=nEdof,solver=lin_solver,
                         springs=None)
    # assemble right hand side
    # forces due to heat expansion per element
    if ndim == 2:
        fTe = _fk_heatexp_2d(xe=xe,
                             c=cs,
                             a=np.eye(ndim),
                             DeltaT=T[TedofMat][:,:,0])
    elif ndim == 3:
        fTe = _fk_heatexp_3d(xe=xe,
                             c=cs,
                             a=np.eye(ndim),
                             DeltaT=T[TedofMat][:,:,0])
    fTe = fTe * a[:,None]
    # assemble
    fT = np.zeros(f.shape)
    np.add.at(fT[:,0],
              EedofMat.flatten(),
              fTe.flatten())
    # assemble completely
    rhsE = assemble_rhs(f0=f+fT,
                        solver=lin_solver)
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE,solver=lin_solver,
                 free=freeE,fixed=fixedE)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[freeE, :], fact, precond, = solve_lin(K=KE, rhs=rhsE[freeE],
                                            solver=lin_solver,
                                            preconditioner=preconditioner)
    #print(u[1::2].max())
    #np.savetxt("surface-displacements.csv",
    #           u[np.arange(0,2*(nelx+1)*(nely+1),2*(nely+1))+1,0])
    #
    if export:
        export_vtk(filename=file+"T"+str(ndim),
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=T)
        export_vtk(filename=file+"E-"+str(ndim),
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=1/a,
                   u=u,f=f+fT)
    return

if __name__ == "__main__":
    #
    nelx=60
    nely=20
    nelz=10
    #
    import sys
    if len(sys.argv)>1:
        nelx = int(sys.argv[1])
    if len(sys.argv)>2:
        nely = int(sys.argv[2])
    if len(sys.argv)>3:
        nelz = int(sys.argv[2])
    #
    fem_heat_expansion(nelx=nelx,nely=nely,nelz=nelz)
