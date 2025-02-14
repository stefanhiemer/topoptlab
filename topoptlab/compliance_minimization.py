from os.path import isfile
from os import remove
import logging 
from functools import partial
#
import numpy as np
from scipy.sparse import coo_matrix,coo_array
from scipy.sparse.linalg import spsolve,factorized
from scipy.ndimage import convolve
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# functions to create filters
from topoptlab.filters import assemble_matrix_filter,assemble_convolution_filter,assemble_helmholtz_filter
# default application case that provides boundary conditions, etc.
from topoptlab.example_cases import mbb_2d
# set up finite element problem
from topoptlab.fem import update_indices
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
from topoptlab.elements.poisson_2d import lk_poisson_2d
# constrained optimizers
from topoptlab.optimality_criterion import oc_top88
from topoptlab.mma_utils import update_mma 
from topoptlab.objectives import compliance
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         nelz=None,
         filter_mode="matrix",
         bcs=mbb_2d,
         el_flags=None,
         solver="oc", nouteriter=2000, ninneriter=15,
         display=True,export=True,write_log=True,
         debug=0):
    """
    Topology optimization workflow with the SIMP method based on 
    the default direct solver of scipy sparse.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
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
    nelz : int or None
        number of elements in z direction. Only important if ndim is 3.
    filter_mode : str
        indicates how filtering is done. Possible values are "matrix" or 
        "helmholtz". If "matrix", then density/sensitivity filters are 
        implemented via a sparse matrix and applied by multiplying 
        said matrix with the densities/sensitivities. 
    bcs : str or function
        returns the boundary conditions
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements. 
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
    solver: str
        solver options which are "oc", "mma" and "gcmma" for the optimality 
        criteria method, the method of moving asymptotes and the globally 
        covergent method of moving asymptotes.
    nouteriter: int 
        number of TO iterations
    ninneriter: int
        number of inner iterations for GCMMA
    display : bool
        if True, plot design evolution to screen
    export : bool
        if True, export design as vtk file.
    write_log : bool
        if True, write a log file and display results to command line.
    debug : bool
        if True, print extra output for debugging.

    Returns
    -------
    None.

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if write_log:
        # check if log file exists and if True delete
        if isfile("topopt.log"):
            remove("topopt.log")
        logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler("topopt.log"),
                            logging.StreamHandler()])
        #
        logging.info(f"minimum compliance problem with {solver}")
        logging.info(f"number of spatial dimensions: {ndim}")
        if ndim == 2:
            logging.info(f"elements: {nelx} x {nely}")
        elif ndim == 3:
            logging.info(f"elements: {nelx} x {nely} x {nelz}")
        logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
        logging.info("filter: " + ["Sensitivity based", 
                                   "Density based",
                                   "Haeviside Guest",
                                   "Haeviside complement Sigmund 2007",
                                   "Haeviside eta projection",
                                   "Volume Preserving eta projection",
                                   "No filter"][ft])
        logging.info(f"filter mode: {filter_mode}")
    # total number of design variables/elements
    if ndim == 2:
        n = nelx * nely
    elif ndim == 3:
        n = nelx * nely * nelz
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(n, dtype=float,order='F')
    xold = x.copy()
    xTilde = x.copy()
    xPhys = x.copy()
    # initialize solver
    if solver=="oc":
        # must be initialized to use the NGuyen/Paulino OC approach
        g = 0  
    elif solver == "mma":
        # number of constraints.
        m = 1 
        # lower and upper bound for densities
        xmin = np.zeros((x.shape[0],1))
        xmax = np.ones((x.shape[0],1))
        # densities of two previous iterations
        xold1 = x.copy() 
        xold2 = x.copy()
        # initial lower and upper asymptotes
        low = np.ones((x.shape[0],1))
        upp = np.ones((x.shape[0],1))
        #
        a0 = 1.0 
        a = np.zeros((m,1)) 
        c = 10000*np.ones((m,1))
        d = np.zeros((m,1))
        move = 0.2
    elif solver == "gcmma":
        # number of constraints.
        m = 1 
        # lower and upper bound for densities
        xmin = np.zeros((x.shape[0],1))
        xmax = np.ones((x.shape[0],1))
        # densities of two previous iterations
        xold1 = x.copy() 
        xold2 = x.copy()
        # lower and upper asymptotes
        low = np.ones((x.shape[0],1))
        upp = np.ones((x.shape[0],1))
        #
        a0 = 1.0 
        a = np.zeros((m,1)) 
        c = 10000*np.ones((m,1))
        d = np.zeros((m,1))
        move = 0.2
        #
        epsimin = 0.0000001
        raa0 = 0.01
        raa = 0.01*np.ones((m,1))
        raa0eps = 0.000001
        raaeps = 0.000001*np.ones((m,1))
    else:
        raise ValueError("Unknown solver: ", solver)
    # Max and min Young's modulus
    Emin = 1e-9
    Emax = 1.0
    # get element stiffness matrix
    if ndim == 2:
        KE = lk_linear_elast_2d() #lk_poisson_2d()#
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3 
        n_ndof = int(KE.shape[0]/4)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*n_ndof
    elif ndim == 3:
        KE = lk_linear_elast_3d()
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3 
        n_ndof = int(KE.shape[0]/8)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*(nelz+1)*n_ndof
    # el. indices
    el = np.arange(n)
    # element degree of freedom matrix plus some helper indices
    if ndim == 2:
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,nelz=nelz,
                                                   nnode_dof=n_ndof)
    elif ndim == 3:
        edofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,nelz=nelz,
                                                   nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK = np.tile(edofMat,KE.shape[0]).flatten()
    jK = np.repeat(edofMat,KE.shape[0]).flatten()
    # function to convert densities, etc. to images/voxels for plotting or the
    # convolution filter.
    if ndim == 2:
        mapping = partial(map_eltoimg,
                          nelx=nelx,nely=nely)
    elif ndim == 3:
        mapping = partial(map_eltovoxel,
                          nelx=nelx,nely=nely,nelz=nelz)
    # prepare functions to invert this mapping if we use the convolution filter
    if filter_mode == "convolution" and ndim == 2:
        invmapping = partial(map_imgtoel,
                             nelx=nelx,nely=nely)
    elif filter_mode == "convolution" and ndim == 3:
        invmapping = partial(map_voxeltoel,
                             nelx=nelx,nely=nely,nelz=nelz)
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    if filter_mode == "matrix":
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,nelz=nelz,
                                      rmin=rmin,el=el,ndim=ndim)
    elif filter_mode == "convolution":
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz, 
                                           rmin=rmin,
                                           mapping=mapping,
                                           invmapping=invmapping)
    elif filter_mode == "helmholtz" and ft in [0,1]:
        KF,TF = assemble_helmholtz_filter(nelx=nelx,nely=nely,nelz=nelz,
                                          rmin=rmin,ndim=ndim,
                                          el=el,n1=n1,n2=n2,n3=n3,n4=n4)
        # LU decomposition. returns a function for solving, not the matrices
        lu_solve = factorized(KF)
    # BC's and support
    u,f,fixed,free = bcs(nelx=nelx,nely=nely,nelz=nelz,
                         ndof=ndof)
    # get rid of fixed degrees of freedom from stiffness matrix 
    mask = ~(np.isin(iK,fixed) | np.isin(jK,fixed))
    iK = update_indices(iK, fixed, mask)
    jK = update_indices(jK, fixed, mask)
    ndof_free = ndof - fixed.shape[0]
    # passive elements 
    if display:
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        if ndim == 2:
            fig,ax = plt.subplots(1,1)
            im = ax.imshow(mapping(-xPhys), cmap='gray',
                           interpolation='none', norm=Normalize(vmin=-1, vmax=0))
            plotfunc = im.set_array
        elif ndim == 3:
            raise NotImplementedError("Plotting in 3D not yet implemented.")
            fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
            im = ax.voxels(mapping(np.ones(xPhys.shape,dtype=bool)),
                           facecolors = -xPhys, 
                           cmap='gray', edgecolor=None,
                           norm=Normalize(vmin=-1, vmax=0))
            plotfunc = im[0].set_facecolors
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        ax.axis("off")
        fig.show()
    # initialize arrays for gradients
    dc = np.zeros(x.shape[0],order="F")
    dv = np.ones(x.shape[0],order="F")
    # optimization loop
    for loop in np.arange(nouteriter):
        # solve FEM, calculate obj. func. and gradients.
        # for 
        if solver in ["oc","mma"] or\
           (solver in ["gcmma"] and ninneriter==0) or\
           loop==0:
            # Setup and solve FE problem
            sK = (KE.flatten()[:,None]*(Emin+(xPhys)
                  ** penal*(Emax-Emin))).flatten(order='F')[mask]
            K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
            #K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
            # Remove constrained dofs from matrix
            #K = K[free, :][:, free]
            # Solve system(s)
            if u.shape[1] == 1:
                u[free, 0] = spsolve(K, f[free, 0])
            else:
                u[free, :] = spsolve(K, f[free, :])
            # Objective and objective gradient
            obj = 0
            dc[:] = np.zeros(x.shape[0])
            for i in np.arange(f.shape[1]):
                obj,dc[:] = compliance(xPhys=xPhys,u=u[:,i],
                                       KE=KE,edofMat=edofMat,
                                       Amax=Emax,Amin=Emin,penal=penal,
                                       obj=obj,dc=dc)
                if debug:
                    print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                           loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        # Constraints and constraint gradients
        dv[:] = np.ones(x.shape[0])
        if debug:
            print("Pre-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dv: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dv)))
            #print(dc)
        # Sensitivity filtering:
        if ft == 0 and filter_mode == "matrix":
            dc[:] = np.asarray((H*(x*dc))[None].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
            #dc[:] = dc[:] = H @ (dc*x) / Hs / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "convolution":
            dc[:] = invmapping(convolve(mapping(dc/hs),
                               h,
                               mode="constant",
                               cval=0)) / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "helmholtz":
            dc[:] = TF.T @ lu_solve(TF@(dc*xPhys))/np.maximum(0.001, x)
        elif ft == 1 and filter_mode == "matrix":
            dc[:] = np.asarray(H*(dc[None].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[None].T/Hs))[:, 0]
            #dc[:] = H @ (dc/Hs)
            #dv[:] = H @ (dv/Hs)
        elif ft == 1 and filter_mode == "convolution":
            dc[:] = invmapping(convolve(mapping(dc/hs),
                                        h,
                                        mode="constant",
                                        cval=0))
            dv[:] = invmapping(convolve(mapping(dv/hs),
                                        h,
                                        mode="constant",
                                        cval=0))
        elif ft == 1 and filter_mode == "helmholtz":
            dc[:] = TF.T @ lu_solve(TF@dc)
            dv[:] = TF.T @ lu_solve(TF@dv)
        elif ft == -1:
            pass
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, max. dc: {1:.10f}, min. dv: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dv)))
        # density update by solver
        xold[:] = x
        # optimality criteria
        if solver=="oc":
            (x[:], g) = oc_top88(x, volfrac, dc, dv, g, el_flags)
        # method of moving asymptotes, implementation by Arjen Deetman
        elif solver=="mma":
            xval = x.copy()[None].T
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = update_mma(x,xold1,xold2,xPhys,obj,dc,dv,loop,
                                                                     m,xmin,xmax,low,upp,a0,a,c,d,move)
            xold2 = xold1.copy()
            xold1 = xval.copy()
            x = xmma.copy().flatten()
        #
        if debug:
            print("Post Density Update: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1 and filter_mode == "matrix":
            xPhys[:] = np.asarray(H*x[None].T/Hs)[:, 0]
            #xPhys[:] = H @ x / Hs
        elif ft == 1 and filter_mode == "convolution":
            xPhys[:] = invmapping(convolve(mapping(x),
                                  h,
                                  mode="constant",
                                  cval=0)) / hs
        elif ft == 1 and filter_mode == "helmholtz":
            xPhys[:] = TF.T @ lu_solve(TF@x)
        elif ft == -1:
            xPhys[:]  = x
        if debug:
            print("Post Density Filter: it.: {0}, med. x.: {1:.10f}, med. xTilde: {2:.10f}, med. xPhys: {3:.10f}".format(
                   loop, np.median(x),np.median(xTilde),np.median(xPhys)))
        # Compute the change by the inf. norm
        change = np.abs(x-xold).max()
        # Plot to screen
        if display:
            if ndim == 2:
                plotfunc(mapping(-xPhys))
            elif ndim == 3:
                im.set_array(mapping(-xPhys))
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if write_log: 
            logging.info("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(
                         loop+1, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break
    #
    if display:
        plt.show()
        input("Press any key...")
    #
    if export:
        export_vtk(filename="topopt", 
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,x=x, 
                   u=u,f=f,volfrac=volfrac)
    return x, obj 