# SPDX-License-Identifier: GPL-3.0-or-later
# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from typing import Callable, Union
from cProfile import Profile
#
import numpy as np
from scipy.sparse import coo_array
#
from matplotlib import colors
import matplotlib.pyplot as plt
#
from topoptlab.material_interpolation import simp, simp_dx,\
                                             bound_interpol, bound_interpol_dx
#
from topoptlab.filter.matrix_filter import assemble_matrix_filter
#
from topoptlab.fem import create_matrixinds,assemble_matrix,apply_bc
from topoptlab.example_bc.lin_elast import mbb_2d,mbb_3d
from topoptlab.example_bc.heat_conduction import heatplate_2d
# element related things
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d
from topoptlab.elements.poisson_3d import lk_poisson_3d
#
from topoptlab.optimizer.optimality_criterion import oc_top88
from topoptlab.optimizer.mma_utils import mma_defaultkws
from mmapy import mmasub
#
from topoptlab.solve_linsystem import solve_lin
# MAIN DRIVER
def main(nelx: int, nely: int, nelz: Union[None,int],
         Es: np.ndarray,
         nus: np.ndarray,
         volfracs: np.ndarray,
         penal: float,
         rmin: float, ft: int,
         bcs: Callable,
         lk: Callable,
         optimizer: str = "mma",
         name: str = "heatplate"):
    """
    Topology optimization for maximum stiffness with a combination of 
    Hashin Shtrikman interpolation and SIMP.
    the default direct solver of scipy sparse.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : None or int
        number of elements in z direction.
    Es : list of floats
        list of Young's moduli or heat conductivities.
    nus : list of floats
        list of Poisson ratios. Ignored for scalar field.
    volfracs : np.ndarray
        volume fraction for solid, then each phase.
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
    # initialize profiling
    profiler = Profile() 
    profiler.enable()
    #
    nmaterials = len(Es)
    #
    if nelz is None:
        ndim = 2
        create_edofMat = create_edofMat2d
    else:
        ndim = 3
        create_edofMat = create_edofMat3d
    print(f"minimum compliance problem with {optimizer}")
    print("elements: "+" x ".join([f"{nelx}",f"{nely}",f"{nelz}"][:ndim]))
    print("volfracs: " + np.array2string(np.asarray(volfracs), 
                                         precision=4, 
                                         floatmode="fixed")  +\
          ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("filter method: " + ["Sensitivity based","Density based"][ft])
    # total number of design elements
    n_mat = 2
    n = np.prod([nelx,nely,nelz][:ndim])
    n_design = n*n_mat
    # 
    eps=1e-9
    # Allocate design variables (as array), initialize and allocate sens.
    x=np.array([volfracs[0],volfracs[1]/volfracs[0]])[None,:]*\
      np.ones((n,n_mat),dtype=float,order="F")
    xold=x.copy()
    xPhys=x.copy()
    if optimizer == "mma":
        optimizer_kw = mma_defaultkws(n_constr=n_mat,
                                      n=n_design)
        optimizer_kw["move"] = 0.1
        xold1 = x.reshape((-1,1), order="F").copy()
        xold = xold1.copy()
    # must be initialized to use the NGuyen/Paulino OC approach
    elif optimizer == "oc":
        g=0 
    # fetch element stiffness matrix
    KE = lk()
    # dofs:
    n_ndof = int(KE.shape[-1]/2**ndim)
    ndof = n_ndof * np.prod( np.array([nelx,nely,nelz][:ndim])+1 )
    # import the correct hashin shtrikman bounds
    if n_ndof == 1 and ndim==3 and nmaterials > 1:
        from topoptlab.bounds.hashin_shtrikman_3d import (conductivity_nary_low as low, 
                                                          conductivity_nary_low_dx as low_dx, 
                                                          conductivity_nary_upp as upp,
                                                          conductivity_nary_upp_dx as upp_dx, 
                                                          )
        bd_kws = {"ks": Es}
    elif n_ndof == 1 and ndim==2 and nmaterials == 2:
        from topoptlab.bounds.hashin_shtrikman_3d import (conductivity_binary_low as low, 
                                                          conductivity_binary_low_dx as low_dx, 
                                                          conductivity_binary_upp as upp,
                                                          conductivity_binary_upp_dx as upp_dx,
                                                         )
        bd_kws = {"kmin": Es.min(), "kmax": Es.max()}
    elif n_ndof == ndim and ndim==3 and nmaterials > 1:
        from topoptlab.bounds.hashin_shtrikman_3d import (emod_nary_low as low, 
                                                          emod_nary_low_dx as low_dx, 
                                                          emod_nary_upp as upp,
                                                          emod_nary_upp_dx as upp_dx,
                                                         )
    elif n_ndof == ndim and ndim==2 and nmaterials == 2:
        from topoptlab.bounds.hashin_shtrikman_3d import (emod_binary_low as low,
                                                          emod_binary_low_dx as low_dx, 
                                                          emod_binary_upp as upp,
                                                          emod_binary_upp_dx as upp_dx,
                                                          )
    # set the keywords for the bounds
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
        if n_mat == 1:
            im = ax.imshow(-xPhys.reshape((nely,nelx),order="F"), cmap='gray',
                           interpolation='none',
                           norm=colors.Normalize(vmin=-1,vmax=0))
        elif n_mat < 4:
            # blue
            blue = np.array([0., 0., 1.])[None,None,:]
            # red
            red = np.array([1., 0., 0.])[None,None,:] 
            #
            colors = (1-xPhys[:,0]).reshape((nely,nelx,1),order="F")*np.ones((1,1,3))+\
                     (xPhys[:,0]*xPhys[:,1]).reshape((nely,nelx,1),order="F")*blue+\
                     (xPhys[:,0]*(1-xPhys[:,1])).reshape((nely,nelx,1),order="F")*red
            #
            im = ax.imshow(colors,
                           interpolation='none')
            
        else:
            raise NotImplementedError("")
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        fig.show()
    # Set loop counter and gradient vectors
    loop,change=0,1
    dc = np.ones((n, n_mat), order="F")
    dconstrs = np.ones((n, n_mat**2), order="F")
    ce = np.ones(n,order="F")
    while change>0.01 and loop<2000:
        loop=loop+1
        # Setup and solve FE problem
        # interpolate material properties
        scale = simp(xPhys=xPhys[:,0],penal=penal, eps=eps)
        scale_dx = simp_dx(xPhys=xPhys[:,0],penal=penal, eps=eps)
        E_hs = bound_interpol(xPhys=xPhys[:,1:], 
                              w = 0.05,
                              bd_low = low, 
                              bd_upp = upp,
                              bd_kws = bd_kws)[:,0]
        E_hs_dx = bound_interpol_dx(xPhys=xPhys[:,1:], 
                                    w = 0.05,
                                    bd_low = low, 
                                    bd_upp = upp,
                                    bd_low_dx = low_dx, 
                                    bd_upp_dx = upp_dx,
                                    bd_kws = bd_kws)
        #
        sK=(KE.flatten()[:,None]*E_hs*scale).flatten(order='F')
        K = coo_array((sK,(iK,jK)),shape=(ndof,ndof)).tocsr()
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
        obj=( E_hs*scale*ce ).sum()
        dc[:]=(-1)*ce[:,None]*np.column_stack((E_hs*scale_dx, 
                                               E_hs_dx*scale[:,None]))
        dconstrs = np.column_stack([np.ones(n),
                                    np.zeros(n),
                                    xPhys[:, 1],
                                    xPhys[:, 0],])
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H@(x*dc))/Hs) / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H@(dc/Hs))
            dconstrs[:] = np.asarray(H@(dconstrs/Hs))
        # reverse reshaping for filtering
        dc = dc.reshape((n,n_mat),order="F")
        dconstrs = dconstrs.reshape((n_design,n_mat),order="F")
        # optimality criteria
        if optimizer == "oc":
            raise NotImplementedError("OC not yet done for multimaterial.")
            xold[:]=x
            x[:],g=oc_top88(x=x,volfrac=volfrac,
                            dc=dc,dv=dconstrs,
                            g=g,el_flags=None)
        elif optimizer == "mma":
            #
            constrs = np.array([[xPhys[:,0].mean() - volfracs[0]], 
                                [(xPhys[:,0]*xPhys[:,1]).mean() - volfracs[1]]])
            xval = x.reshape((-1,1), order="F").copy()
            #
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,l,u = \
                                              mmasub(m=optimizer_kw["nconstr"],
                                                     n=n_design,
                                                     iter=loop,
                                                     xval=xval,
                                                     xold1=xold,
                                                     xold2=xold1,
                                                     f0val=obj,
                                                     df0dx=dc.reshape((-1,1), order="F"),
                                                     fval=constrs,
                                                     dfdx=dconstrs.T,
                                                     **optimizer_kw)
            xold1 = xold.copy()
            xold = xval.copy()
            x[:] = xmma.reshape((n, n_mat), order="F")
        # filter design variables
        if ft==0:
            xPhys[:]=x
        elif ft==1:
            xPhys[:]=np.asarray(H@x/Hs)
        elif ft==-1:
            xPhys[:]=x
        # compute the change by the inf. norm
        change=np.abs(x.reshape((-1,1), order="F")-xold).max()
        # Plot to screen
        if ndim == 2:
            if n_mat == 1:
                im.set_array(-xPhys.reshape((nely,nelx),order="F"))
            elif n_mat < 5:
                #
                colors = (1-xPhys[:,0]).reshape((nely,nelx,1),order="F")*np.ones((1,1,3))+\
                         (xPhys[:,0]*xPhys[:,1]).reshape((nely,nelx,1),order="F")*blue+\
                         (xPhys[:,0]*(1-xPhys[:,1])).reshape((nely,nelx,1),order="F")*red
                im.set_array(colors)
            #
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.10f} vol.: {2}, ch.: {3:.10f}".format(\
                    loop,
                    obj,
                    np.array2string(np.cumprod(xPhys,axis=1).mean(axis=0), 
                                    precision=4, floatmode="fixed"),
                    change))
    # Make sure the plot stays and that the shell remains
    plt.show()
    # finish profiling
    profiler.disable()
    profiler.dump_stats("mbb_scratch.prof")
    input("Press any key...")
    return
# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = nelx
    nelz=None
    volfracs = np.array([0.5,0.4])
    rmin = 1.2/40 * nelx
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
    if len(sys.argv) > 4:
        volfracs = np.array([float(v) for v in sys.argv[4].split(",")],
                            dtype=float)
    if len(sys.argv)>5: 
        rmin   =float(sys.argv[5])
    if len(sys.argv)>6: 
        penal  =float(sys.argv[6])
    if len(sys.argv)>7: 
        ft     =int(sys.argv[7])
    #
    if nelz is None:
        bcs = heatplate_2d
        lk = lk_poisson_2d#lk_linear_elast_2d
    else:
        bcs = mbb_3d
        lk = lk_poisson_3d#lk_linear_elast_3d
    #
    main(nelx=nelx,nely=nely,nelz=nelz,
         Es = np.array([1e0,1e-1]),
         nus = np.array([1/3,1/3]),
         volfracs=volfracs,
         penal=penal,rmin=rmin,ft=ft,
         bcs=bcs, lk=lk)
