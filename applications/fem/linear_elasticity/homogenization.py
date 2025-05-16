#
import numpy as np
from scipy.linalg import solve
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.bilinear_quadrilateral import apply_pbc as apply_pbc2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.trilinear_hexahedron import apply_pbc as apply_pbc3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d,lf_strain_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d, lf_strain_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk

# MAIN DRIVER
def fem_homogenization(nelx, nely, nelz=None,
                       xPhys=None, penal=3, 
                       Emax=1.0, Emin=1e-9, nu=0.3,
                       lin_solver="scipy-direct", preconditioner=None,
                       assembly_mode="full", l=1.,
                       file="homogenization",
                       export=False):
    """
    Run a single finite element simulation on a regular grid performing linear 
    homogenization along the lines of 
    
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
    # get element stiffness matrix, nodal forces and  element of freedom matrix
    if ndim == 2:
        Ke = lk_linear_elast_2d(E=1.0, nu=nu,l=l)
        fe = []
        eps = np.eye(3)
        for i in range(int((ndim**2 + ndim) /2)):
            fe.append(lf_strain_2d(eps[i],E=1.0, nu=nu,l=l))
        fe = np.column_stack(fe)
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        nd_ndof = int(Ke.shape[0]/4)
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=nd_ndof)
        # apply pbc
        edofMat = apply_pbc2d(edofMat=edofMat, pbc=(True,True), nelx=nelx, nely=nely, 
                              nnode_dof=nd_ndof)
    elif ndim == 3:
        Ke = lk_linear_elast_3d(E=1.0, nu=nu,l=l)
        fe = []
        eps = np.eye(6)
        for i in range(int((ndim**2 + ndim) /2)):
            fe.append(lf_strain_3d(eps[i],E=1.0, nu=nu,l=l))
        fe = np.column_stack(fe)
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
    # find the elemental affine displacement field
    if ndim == 2:
        fixed = np.array([0,1,3])
    else:
        fixed = np.array([0,1,2,4,5,7,8])
    free = np.setdiff1d(np.arange(Ke.shape[-1]), fixed)
    u0 = np.zeros(fe.shape)
    u0[free] = solve(Ke[free,:][:,free],fe[free,:],assume_a="sym")
    eigval,eigvec = np.linalg.eigh(Ke[free,:][:,free])
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,mode=assembly_mode)
    # BC's and support
    u,f,fixed,free,_ = bc_singlenode(nelx=nelx,nely=nely,nelz=nelz,
                                     ndof=ndof)
    # interpolate material properties
    scale = (Emin+(xPhys)** penal*(Emax-Emin))
    Kes = Ke[None,:,:]*scale[:,None,None]
    fes = fe[None,:,:]*scale[:,None,None]
    # create entries of stiffness matrix
    if assembly_mode == "full":
        #sK = (KE.flatten()[:,None]*scale).flatten(order='F')
        sK = Kes.flatten()
    # assemble stiffness matrix
    KE = assemble_matrix(sK=sK,iK=iK,jK=jK,
                         ndof=ndof,solver=lin_solver,
                         springs=None)
    # assemble forces
    np.add.at(f,
              edofMat,
              fes)
    # assemble completely
    rhs = assemble_rhs(f0=f,
                       solver=lin_solver)
    # apply boundary conditions to matrix
    KE = apply_bc(K=KE, solver=lin_solver,
                  free=free, fixed=fixed)
    # solve linear system. fact is a factorization and precond a preconditioner
    u[free, :], fact, precond, = solve_lin(K=KE, rhs=rhs[free], 
                                           solver=lin_solver,
                                           preconditioner=preconditioner)
    # calculate effective elastic tensor
    CH = np.zeros((fe.shape[-1], fe.shape[-1]))
    cellVolume = np.prod(l)
    for i in range(fe.shape[-1]):
        for j in range(fe.shape[-1]):
            sumLambda = (((u0[None,:, i] - u[edofMat, i]) @ Kes) * \
                         (u0[None, :, j] - u[edofMat,j])).sum(axis=1)
            # Homogenized elasticity tensor
            CH[i, j] = (1 / cellVolume) * np.sum(sumLambda)
    print('--- Homogenized elasticity tensor ---')
    print(CH)
    #
    if export:
        export_vtk(filename=file,
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,
                   u=u,f=f)
    return

def bc_singlenode(nelx,nely,ndof,nelz=None,**kwargs):
    """
    Fix only first node. Typically used for homogenization or similar.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    ndof : int
        number of degrees of freedom.
    nelz : int or None
        number of elements in z direction.
        
    Returns
    -------
    u : np.ndarray
        array of zeros for state variable (displacement, temperature) to be
        filled of shape (ndof).
    f : np.ndarray
        array of zeros for state flow variables (forces, flow).
    fixed : np.ndarray
        indices of fixed dofs (nfixed).
    free : np.ndarray
        indices of free dofs (ndofs - nfixed).

    """
    if nelz is None:
        ndim=2
    else:
        ndim=3
    #
    dofs = np.arange(ndof)
    # Solution and RHS vectors
    f = np.zeros((ndof, int((ndim**2 + ndim) /2)))
    u = np.zeros((ndof, int((ndim**2 + ndim) /2)))
    # fix first node
    if nelz is None:
        fixed = dofs[:int(ndof/((nelx+1)*(nely+1)*4))]
    else:
        fixed = dofs[:int(ndof/((nelx+1)*(nely+1)*(nelz+1)*8))]
    return u,f,fixed,np.setdiff1d(dofs,fixed),None

if __name__ == "__main__":
    #
    nelx = 2
    nely = 2
    nelz = None
    #
    np.random.seed(0)
    if nelz is None:
        #xPhys = np.random.rand(nelx*nely)
        xPhys = np.eye(2)
    else:
        xPhys = np.random.rand(nelx*nely*nelz)
    #
    import sys
    if len(sys.argv)>1: 
        nelx = int(sys.argv[1])
    if len(sys.argv)>2: 
        nely = int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz = int(sys.argv[3])
    #
    fem_homogenization(nelx=nelx, nely=nely, nelz=nelz,xPhys=xPhys)
