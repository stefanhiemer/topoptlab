# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# modifications by Stefan Hiemer (December 2025)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely):
    """
    Minimal FEM example used during teaching demonstrating what happens to 
    the full material mbb-beam.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction. 

    Returns
    -------
    None.

    """
    print("FEM")
    print("ndes: " + str(nelx) + " x " + str(nely))
    # Young's modulus
    Emax=1.0
    # fetch element stiffness matrix
    KE = lk()
    # # dofs:
    ndof = int(KE.shape[-1]/4) *(nelx+1)*(nely+1)
    # FE: Build the index vectors for the for coo matrix format.
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                               2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # Construct the index pointers for the coo format
    iK = np.tile(edofMat,KE.shape[-1]).flatten()
    jK = np.repeat(edofMat,KE.shape[-1]).flatten() 
    # BC's and support
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed = np.hstack((np.arange(0,2*(nely+1),2), # symmetry
                       np.array([2*(nelx+1)*(nely+1)-1]))) # fixation bottom right
    free=np.setdiff1d(dofs,fixed)
    # Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
    # Set load
    f[1,0]=-1
    # Setup and solve FE problem
    sK=Emax*(KE.flatten()[:,None]*np.ones(el.max()+1)).flatten(order='F')
    K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
    # Remove constrained dofs from matrix
    K = K[free,:][:,free]
    # Solve system
    u[free,0]=spsolve(K,f[free,0])
    #
    from topoptlab.output_designs import export_vtk
    export_vtk(filename="minimal-fem.vtk",
              nelx=nelx,nely=nely,nelz=None,
               xPhys=np.ones(el.max()+1),
               u=u,f=f)
    return 
#element stiffness matrix
def lk():
    """
    Create element stiffness matrix for 2D linear elasticity equation with
    bilinear quadrilateral elements in plane stress. Taken from the 88 line code.

    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.

    """
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,
                -1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE 
# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx=60
    nely=20 
    ft=1 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: 
        nelx   =int(sys.argv[1])
    if len(sys.argv)>2: 
        nely   =int(sys.argv[2])
    if len(sys.argv)>3: 
        volfrac=float(sys.argv[3])
    main(nelx,nely)
