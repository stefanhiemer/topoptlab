from os.path import isfile
from os import remove
import logging
#
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import threepointbending_2d,xcenteredbeam_2d
from topoptlab.example_bc.heat_conduction import rectangle_2d
# different elements/physics
from topoptlab.stiffness_tensors import isotropic_2d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d
from topoptlab.elements.heatexpansion_2d import _fk_heatexp_2d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_heat_expansion(nelx, nely, nelz=None,
                       xPhys=None, penal=3, 
                       Emax=1.0, Emin=1e-9, nu=0.3, 
                       kmax=1.0, kmin=1e-9,
                       alpha=0.05,
                       lin_solver="cvxopt-cholmod", preconditioner=None,
                       assembly_mode="full",
                       bcs=[xcenteredbeam_2d, # threepointbending_2d
                            rectangle_2d],
                       file="rod_heat-expansion",
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
    xPhys : np.ndarray
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
    kmax : float
        (maximum) Yheat conductivity. If xPhys is None, all elements take this 
        value.
    kmin : float
        minimum heat conductivity for the modified SIMP approach.
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
        raise NotImplementedError()
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
    elif ndim == 3:
        n = nelx * nely * nelz
    #
    if xPhys is None:
        xPhys = np.ones(n, dtype=float,order='F')
    # get element stiffness matrix and element of freedom matrix
    if ndim == 2:
        KE = lk_linear_elast_2d()
        KT = lk_poisson_2d()
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nE_ndof = int(KE.shape[0]/4)
        nT_ndof = int(KT.shape[0]/4)
        # number of degrees of freedom
        nEdof = (nelx+1)*(nely+1)*nE_ndof
        nTdof = (nelx+1)*(nely+1)*nT_ndof
        # element degree of freedom matrix plus some helper indices
        EedofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                    nnode_dof=nE_ndof)
        TedofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                    nnode_dof=nT_ndof)
    elif ndim == 3:
        KE = lk_linear_elast_3d()
        KT = lk_poisson_3d()
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nE_ndof = int(KE.shape[0]/8)
        nT_ndof = int(KT.shape[0]/8)
        # number of degrees of freedom
        nEdof = (nelx+1)*(nely+1)*(nelz+1)*nE_ndof
        nTdof = (nelx+1)*(nely+1)*(nelz+1)*nT_ndof
        # element degree of freedom matrix plus some helper indices
        EedofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                    nelz=nelz,
                                                    nnode_dof=nE_ndof)
        TedofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                    nelz=nelz,
                                                    nnode_dof=nT_ndof)
    # Construct the index pointers for the coo format
    iKE,jKE = create_matrixinds(EedofMat,mode=assembly_mode)
    iKT,jKT = create_matrixinds(TedofMat,mode=assembly_mode)
    # BC's and support
    u,f,fixedE,freeE,_ = bcs[0](nelx=nelx,nely=nely,nelz=nelz,
                                ndof=nEdof)
    T,q,fixedT,freeT,_ = bcs[1](nelx=nelx,nely=nely,nelz=nelz,
                                ndof=nTdof)
    # interpolate material properties
    E = (Emin+(xPhys)** penal*(Emax-Emin))
    k = (Emin+(xPhys)** penal*(Emax-Emin))
    # entries of stiffness matrix
    if assembly_mode == "full":
        sKE = (KE.flatten()[:,None]*E).flatten(order='F')
        sKT = (KT.flatten()[:,None]*k).flatten(order='F')
    # Setup and solve temperatur problem
    KT = assemble_matrix(sK=sKT,iK=iKT,jK=jKT,
                         ndof=nTdof,solver=lin_solver,
                         springs=None)
    # assemble right hand side
    rhsT = assemble_rhs(f0=q,solver=lin_solver)
    # apply boundary conditions to matrix
    KT = apply_bc(K=KT,solver=lin_solver,
                 free=freeT,fixed=fixedT)
    # solve linear system. fact is a factorization and precond a preconditioner
    T[freeT, :], fact, precond, = solve_lin(K=KT, rhs=rhsT[freeT], 
                                            solver=lin_solver,
                                            preconditioner=preconditioner)
    #
    KE = assemble_matrix(sK=sKE,iK=iKE,jK=jKE,
                         ndof=nEdof,solver=lin_solver,
                         springs=None)
    # assemble right hand side
    c = E[:,None,None] * isotropic_2d(E=1,nu=nu)[None,:,:]
    # element nodes coordinates in ref. space
    xe = np.array([[[-1.,-1.], 
                    [1.,-1.], 
                    [1.,1.], 
                    [-1.,1.]]]) * np.ones(xPhys.shape)[:,None,None]
    # forces due to heat expansion per element
    fTe = _fk_heatexp_2d(xe=xe,
                         c=c,
                         a=np.eye(ndim)+alpha,
                         DeltaT=T[TedofMat][:,:,0])
    # scale by SIMP interpolation
    fTe = (Emin+(xPhys)**penal*(Emax-Emin))[:,None]*fTe[:]
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
    
    print(u.max(),T.max() * (1+alpha) * nelx)
    #
    if export:
        export_vtk(filename=file+"T",
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=T,f=q)
        export_vtk(filename=file+"E",
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=u,f=f+fT)
    return

if __name__ == "__main__":
    
    fem_heat_expansion(nelx=100, nely=100)
