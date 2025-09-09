# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import tensiletest_2d,tensiletest_3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_tensiletest(nelx, nely, nelz=None,
                    xPhys=None, penal=3, 
                    Emax=1.0, Emin=1e-9, nu=0.3,
                    lin_solver="cvxopt-cholmod", preconditioner=None,
                    assembly_mode="full", l=1.,
                    file="tensiletest",
                    export=True):
    """
    Run a single finite element simulation on a regular grid performing a 
    tensile test pulling along x-direction.

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
    solver : str
        solver for linear systems. Check function lin solve for available 
        options.
    preconditioner : str or None
        preconditioner for linear systems. 
    assembly_mode : str
        whether full or only lower triangle of linear system / matrix is 
        created.
    l : float or tuple of length (ndim) or np.ndarray of shape (ndim)
        side lengths of each element
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
    # 
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
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
        KE = lk_linear_elast_2d(E=1.0, nu=nu, l=l)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nd_ndof = int(KE.shape[0]/4)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*nd_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=nd_ndof)
    elif ndim == 3:
        KE = lk_linear_elast_3d(E=1.0, nu=nu,l=l)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nd_ndof = int(KE.shape[0]/8)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*(nelz+1)*nd_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                   nelz=nelz,
                                                   nnode_dof=nd_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,mode=assembly_mode)
    # BC's and support
    if ndim==2:
        u,f,fixed,free,_ = tensiletest_2d(nelx=nelx,nely=nely,nelz=nelz,
                                          ndof=ndof)
    elif ndim==3:
        u,f,fixed,free,_ = tensiletest_3d(nelx=nelx,nely=nely,nelz=nelz,
                                          ndof=ndof)
    # interpolate material properties
    E = (Emin+(xPhys)** penal*(Emax-Emin))
    # entries of stiffness matrix
    if assembly_mode == "full":
        sK = (KE.flatten()[:,None]*E).flatten(order='F')
    #
    KE = assemble_matrix(sK=sK,iK=iK,jK=jK,
                         ndof=ndof,solver=lin_solver,
                         springs=None)
    # assemble completely
    rhs = f
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE,solver=lin_solver,
                 free=free,fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[free, :], fact, precond, = solve_lin(K=KE, rhs=rhs[free], 
                                           solver=lin_solver,
                                           preconditioner=preconditioner)
    if isinstance(l,float):
        l = tuple([l])*ndim
    if ndim == 2:
        sigma = f.max()*(nely+1)/l[1]
        print("sigma: ", sigma)
        print("eps_xx theory ",sigma/Emax)
        print("eps_yy theory ",sigma/Emax * nu)
        print("eps_xx measured ",np.abs(u[::2]).max()/(nelx*l[0]) )
        print("eps_yy measured",np.abs(u[1::2].max())/(nely*l[1]) )
    elif ndim == 3:
        sigma = f.max() * (nely+1)*(nelz+1)/(nely*nelz * l[1]*l[2])
        print("sigma: ",sigma)
        print("eps_xx theory ",sigma/Emax)
        print("eps_yy theory ",sigma/Emax * nu)
        print("eps_zz theory ",sigma/Emax * nu)
        print("eps_xx measured ",np.abs(u[::3]).max()/ (nelx*l[0]) )
        print("eps_yy measured",np.abs(u[1::3]).max()/ (nely*l[1]) )
        print("eps_zz measured",np.abs(u[2::3]).max()/ (nelz*l[2]) )
    #
    if export:
        export_vtk(filename=file,
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=u,f=f)
    return

if __name__ == "__main__":
    #
    nelx = 100
    nely = 100
    nelz = None
    #
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz = int(sys.argv[3])
    #
    fem_tensiletest(nelx=nelx, nely=nely, nelz=nely)
