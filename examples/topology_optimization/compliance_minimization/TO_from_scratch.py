# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# rewrite with the topoptlab package by Stefan Hiemer (January 2025)
from typing import Callable, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from matplotlib import colors
import matplotlib.pyplot as plt

from topoptlab.filter.matrix_filter import assemble_matrix_filter
from topoptlab.fem import create_matrixinds,assemble_matrix,apply_bc
from topoptlab.example_bc.lin_elast import mbb_2d,mbb_3d
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
from topoptlab.optimizer.optimality_criterion import oc_top88
from topoptlab.solve_linsystem import solve_lin
# MAIN DRIVER
def main(nelx: int, nely: int, nelz: Union[None,int],
         volfrac: float,
         penal: float,
         rmin: float, ft: int,
         bcs: Callable,
         lk: Callable):
    """
    Topology optimization for maximum stiffness with the SIMP method based on
    the default direct solver of scipy sparse.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : None or int
        number of elements in z direction.
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
    bcs : None or callable
        boundary conditions.
    lk : None or callable
        element stiffness matrix.

    Returns
    -------
    None.

    """
    #
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
    else:
        ndim = 3
        create_edofMat = create_edofMat3d
    print("Minimum compliance problem with OC")
    print("elements: "+" x ".join([f"{nelx}",f"{nely}",f"{nelz}"][:ndim]))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # total number of design elements
    n = np.prod([nelx,nely,nelz][:ndim])
    # Max and min stiffness
    Emin=1e-9
    Emax=1.0
    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(n,dtype=float,order="F")
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    # fetch element stiffness matrix
    KE = lk()
    # dofs:
    n_ndof = int(KE.shape[-1]/2**ndim)
    ndof = n_ndof * np.prod( np.array([nelx,nely,nelz][:ndim])+1 )
    # FE: Build the index vectors for the for coo matrix format.
    el = np.arange(n)
    # element degree of freedom matrix plus some helper indices
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,nely=nely,nelz=nelz,
                                             nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat=edofMat, mode="full")
    # assemble filter
    H,Hs = assemble_matrix_filter(rmin=rmin,el=el,nelx=nelx,nely=nely,nelz=nelz)
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx,nely=nely, nelz=nelz,
                                 ndof=ndof)
    # Initialize plot and plot the initial design
    if ndim == 2:
        plt.ion() # Ensure that redrawing is possible
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
    # Set loop counter and gradient vectors
    loop,change=0,1
    dv, dc, ce = np.ones(n), np.ones(n), np.ones(n)
    while change>0.01 and loop<2000:
        loop=loop+1
        # Setup and solve FE problem
        sK=(KE.flatten()[:,None]*(Emin+xPhys**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = apply_bc(K=K,solver="scipy-direct",
                     free=free,fixed=fixed)
        # Solve system
        u[free, :], fact, precond = solve_lin(K=K, rhs=f[free],
                                              solver="scipy-direct",
                                              preconditioner=None)
        # Objective and sensitivity
        ce[:] = (np.dot(u[edofMat,0],KE) * u[edofMat,0]).sum(1)
        #
        obj=( (Emin+xPhys**penal*(Emax-Emin))*ce ).sum()
        dc[:]=(-penal*xPhys**(penal-1)*(Emax-Emin))*ce
        dv[:] = np.ones(n)
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H*(x*dc))[:,None]/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H*(dc[:,None]/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[:,None]/Hs))[:,0]
        # Optimality criteria
        xold[:]=x
        x[:],g=oc_top88(x=x,volfrac=volfrac,dc=dc,dv=dv,g=g,el_flags=None)
        # Filter design variables
        if ft==0:
            xPhys[:]=x
        elif ft==1:
            xPhys[:]=np.asarray(H*x[:,None]/Hs)[:,0]
        # Compute the change by the inf. norm
        change=np.abs(x-xold).max()
        # Plot to screen
        if ndim == 2:
            im.set_array(-xPhys.reshape((nely,nelx),order="F"))
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(\
                    loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    # Make sure the plot stays and that the shell remains
    plt.show()
    input("Press any key...")
    return
# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx=60
    nely=20
    nelz=None
    volfrac=0.5
    rmin=2.4
    penal=3.0
    ft=1 # ft==0 -> sens, ft==1 -> dens
    # put in arguments via command line
    import sys
    if len(sys.argv)>1: 
        nelx   =int(sys.argv[1])
    if len(sys.argv)>2: 
        nely   =int(sys.argv[2])
    if len(sys.argv)>3: 
        nelz   =int(sys.argv[3])
        if nelz==0:
            nelz=None
    if len(sys.argv)>4: 
        volfrac=float(sys.argv[4])
    if len(sys.argv)>5: 
        rmin   =float(sys.argv[5])
    if len(sys.argv)>6: 
        penal  =float(sys.argv[6])
    if len(sys.argv)>7: 
        ft     =int(sys.argv[7])
    #
    if nelz is None:
        bcs = mbb_2d
        lk = lk_linear_elast_2d
    else:
        bcs = mbb_3d
        lk = lk_linear_elast_3d
    #
    main(nelx=nelx,nely=nely,nelz=nelz,
         volfrac=volfrac,penal=penal,rmin=rmin,ft=ft,
         bcs=bcs, lk=lk)
