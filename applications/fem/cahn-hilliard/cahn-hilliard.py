#
import numpy as np
from scipy.linalg import solve
from scipy.sparse.linalg import factorized, eigsh
import matplotlib.pyplot as plt
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.bilinear_quadrilateral import apply_pbc as apply_pbc2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.trilinear_hexahedron import apply_pbc as apply_pbc3d
# different elements/physics
from topoptlab.elements.mass_scalar_2d import lm_mass_2d
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.monomial_scalar_2d import _lm_monomial_2d, lm_cubic_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d, lf_strain_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# boundary condition
from topoptlab.example_bc.lin_elast import singlenode
# output final design to a Paraview readable format
from topoptlab.utils import map_eltoimg
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_cahn(nelx, nely, nelz=None, n_steps=10000,
                       xPhys=None, penal=3,
                       Emax=1.0, Emin=1e-3, nu=1/3,
                       lin_solver="scipy-lu", preconditioner=None,
                       assembly_mode="full", l=1.,
                       file="cahn-hilliard", display=True,
                       export=False, debug=False):
    """
    Run a single finite element simulation on a regular grid solving the 
    Cahn-Hilliard equation.

    Andreassen, Erik, and Casper Schousboe Andreasen. "How to determine
    composite material properties using numerical homogenization."
    Computational Materials Science 83 (2014): 488-495.

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
        side lengths of domain.
    file : str
        name of output files
    export : bool
        if True, export design as vtk file.

    Returns
    -------
    None.

    """
    #
    dt=5.e-3
    gamma=0.5
    mobility=1.0
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # calculate side lengths of elements
    el_sidelengths = l / np.array([nelx,nely,nelz][:ndim])
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
        xe = (el_sidelengths* np.array([[-1,-1],[1,-1],[1,1],[-1,1]])/2) * np.ones(n)[:,None,None]
    elif ndim == 3:
        n = nelx * nely * nelz
        xe = (el_sidelengths* np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                                        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])/2) * np.ones(n)[:,None,None]
    #
    if xPhys is None:
        xPhys = np.ones(n, dtype=float,order='F')
    # get element stiffness matrix, nodal forces and  element of freedom matrix
    if ndim == 2:
        Ke = lk_poisson_2d(l=el_sidelengths)
        Me = lm_mass_2d(l=el_sidelengths)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nd_ndof = int(Ke.shape[0]/4)
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=nd_ndof)
        # apply pbc
        edofMat = apply_pbc2d(edofMat=edofMat, pbc=(True,True), nelx=nelx, nely=nely,
                              nnode_dof=nd_ndof)
    elif ndim == 3:
        Ke = lk_linear_elast_3d(E=1.0, nu=nu,l=el_sidelengths)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nd_ndof = int(Ke.shape[0]/8)
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,
                                                   nelz=nelz,
                                                   nnode_dof=nd_ndof)
        # apply pbc
        edofMat = apply_pbc3d(edofMat=edofMat, pbc=(True,True,True),
                              nelx=nelx, nely=nely, nelz=nelz,
                              nnode_dof=nd_ndof)
    #
    ndof = edofMat.max()+1
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,mode=assembly_mode)
    # interpolate material properties
    scale = (Emin+(xPhys)** penal*(Emax-Emin))
    Kes = Ke[None,:,:]*scale[:,None,None]
    Mes = Me[None,:,:]*scale[:,None,None]
    # create entries of stiffness matrix
    if assembly_mode == "full":
        #sK = (KE.flatten()[:,None]*scale).flatten(order='F')
        sK = Kes.reshape(np.prod(Kes.shape))
        sM = Mes.reshape(np.prod(Mes.shape))
    # assemble stiffness matrix
    KE = assemble_matrix(sK=sK,iK=iK,jK=jK,
                         ndof=ndof,solver=lin_solver,
                         springs=None)
    ME = assemble_matrix(sK=sM,iK=iK,jK=jK,
                         ndof=ndof,solver=lin_solver,
                         springs=None)
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE, solver=lin_solver,
                  free=np.arange(ndof), fixed=np.array([]))
    ME = apply_bc(K=ME, solver=lin_solver,
                  free=np.arange(ndof), fixed=np.array([]))
    #
    if lin_solver == "scipy-lu":
        lu_solve = factorized(ME)
    # initial concentration profile
    c = np.random.rand(ndof) * 0.01
    c = c - c.mean()
    mu = np.zeros(ndof)
    # time integration
    for step in np.arange(n_steps):
        # build the cubic part
        Aes = _lm_monomial_2d(xe=xe,u=c[edofMat],n=3)
        if assembly_mode == "full":
            sA = Aes.reshape(np.prod(Aes.shape))
        AE = assemble_matrix(sK=sA,iK=iK,jK=jK,
                             ndof=ndof,solver=lin_solver,
                             springs=None)
        # chemical potential
        mu[:] = lu_solve(AE@c + gamma * KE@c - ME@c)
        # concentration
        c[:] = lu_solve(ME@c - mobility * dt * KE@mu)
        #
        if ndim == 2 and step % (n_steps // 1000) == 0 and display:
            plt.imshow(map_eltoimg(c,nelx=nelx,nely=nely),
                       cmap='RdBu', origin='lower')
            plt.colorbar(label='Concentration')
            plt.title(f"Step {step}")
            plt.pause(0.001)
            plt.clf()
        print("time.: {0:.10f} min(c).: {1:.10f} max(c).: {2:.10f} volfrac.: {3:.10f}".format(
                     dt*(step+1), c.min(), c.max(), (c[edofMat]*np.prod(el_sidelengths) / 2**ndim).sum() / n ))
    # Final visualization
    if ndim == 2 and display:
        plt.imshow(map_eltoimg(c,nelx=nelx,nely=nely),
                   cmap='RdBu', origin='lower')
        plt.colorbar(label='Concentration')
        plt.title("Final Step")
        plt.show()
    #
    #if export:
    #    export_vtk(filename=file,
    #               nelx=nelx,nely=nely,nelz=nelz,
    #               xPhys=xPhys,
    #               u=u,f=f)
    return



if __name__ == "__main__":
    #
    np.random.seed(0)
    #
    nelx = 128
    nely = 128
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
    fem_cahn(nelx=nelx, nely=nely, nelz=nelz, l=128.)
