# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.heat_conduction import heatplate_2d, heatplate_3d
# different elements/physics
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
#
from topoptlab.material_interpolation import simp
# MAIN DRIVER
def fem_stationary_heatconduction(nelx, 
                                  nely, 
                                  nelz=None,
                                  xPhys=None, 
                                  penal=3, 
                                  kmax=1.0, 
                                  kmin=1e-9, 
                                  lin_solver="cvxopt-cholmod", 
                                  preconditioner=None,
                                  assembly_mode="lower", 
                                  l=1.,
                                  file="stationary_heatconduction",
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
    kmax : float
        (maximum) heat conductivity. If xPhys is None, all elements take this 
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
        lk = lk_poisson_2d
        create_edofMat = create_edofMat2d
        bc = heatplate_2d
    else:
        ndim = 3
        lk = lk_poisson_3d
        create_edofMat = create_edofMat3d
        bc = heatplate_3d
    # 
    if isinstance(l,float):
        l = np.array( [l for i in np.arange(ndim)])
    # total number of design variables/elements
    n = np.array([nelx,nely,nelz][:ndim])
    #
    if xPhys is None:
        xPhys = np.ones((np.prod(n).astype(int),1), 
                        dtype=float, order='F')
    # get element stiffness matrix and element of freedom matrix
    KE = lk(k=1.0, 
            l=l)
    # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
    nd_ndof = int(KE.shape[0]/2**ndim)
    # number of degrees of freedom
    ndof = np.prod(n+1)*nd_ndof
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,
                                             nely=nely,
                                             nelz=nelz,
                                             nnode_dof=nd_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,
                              mode=assembly_mode)
    if assembly_mode == "lower":
        assm_indcs = np.column_stack(np.tril_indices_from(KE))
        assm_indcs = assm_indcs[np.lexsort( (assm_indcs[:,0],assm_indcs[:,1]) )]
    elif assembly_mode == "full":
        pass 
    else:
        raise ValueError("Unknown assembly_mode: ", assembly_mode)
    # BC's and support
    T,f,fixed,free,_ = bc(nelx=nelx,
                          nely=nely,
                          nelz=nelz,
                          ndof=ndof)
    # interpolate material properties
    Kes = kmax*KE[None,:,:]*simp(xPhys=xPhys, 
                                 eps=kmin/kmax, 
                                 penal=penal)[:,:,None]
    # entries of stiffness matrix
    if assembly_mode == "full":
        # this here is more memory efficient than Kes.flatten() as it
        # provides a view onto the original Kes array instead of a copy
        sK = Kes.reshape(np.prod(Kes.shape))
    elif assembly_mode == "lower":
        sK = Kes[:,
                 assm_indcs[:,0],
                 assm_indcs[:,1]].reshape( np.prod(n)*int(KE.shape[-1]/2*(KE.shape[-1]+1)))
    # assemble completely
    KE = assemble_matrix(sK=sK,
                         iK=iK,
                         jK=jK,
                         ndof=ndof,
                         solver=lin_solver,
                         springs=None)
    rhs = f
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE,
                  solver=lin_solver,
                  free=free,fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    T[free, :], fact, precond, = solve_lin(K=KE, 
                                           rhs=rhs[free], 
                                           solver=lin_solver,
                                           preconditioner=preconditioner)
    #
    if export:
        export_vtk(filename=file,
                   nelx=nelx,
                   nely=nely,
                   nelz=nelz,
                   xPhys=xPhys,
                   elem_size=l,
                   u=T,
                   f=f)
    return

if __name__ == "__main__":
    #
    nelx = 10
    nely = 10
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
    fem_stationary_heatconduction(nelx=nelx, nely=nely, nelz=nely)