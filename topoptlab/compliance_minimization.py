from os.path import isfile
from os import remove
import logging
from functools import partial
#
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.ndimage import convolve
#
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
from skimage.measure import marching_cubes
# functions to create filters
from topoptlab.filters import assemble_matrix_filter,assemble_convolution_filter,assemble_helmholtz_filter
# default application case that provides boundary conditions, etc.
from topoptlab.example_bc.lin_elast import mbb_2d
# set up finite element problem
from topoptlab.fem import create_matrixinds
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
# different elements/physics
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.elements.linear_elasticity_3d import lk_linear_elast_3d
# generic functions for solving phys. problem
from topoptlab.fem import assemble_matrix,assemble_rhs,apply_bc
from topoptlab.solve_linsystem import solve_lin
# constrained optimizers
from topoptlab.optimizer.optimality_criterion import oc_top88,oc_mechanism,oc_generalized
from topoptlab.optimizer.mma_utils import update_mma,mma_defaultkws
from topoptlab.objectives import compliance
# output final design to a Paraview readable format
from topoptlab.output_designs import export_vtk
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
# logging related stuff
from topoptlab.log_utils import init_logging


# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft,
         nelz=None,
         filter_mode="matrix",
         lin_solver="scipy-direct", preconditioner=None,
         assembly_mode="full",
         bcs=mbb_2d, 
         obj_func=compliance, obj_kw={},
         el_flags=None,
         optimizer="oc", optimizer_kw = None,
         alpha=None,
         nouteriter=2000, ninneriter=15,
         file="topopt",
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
        number of elements in z direction. If None, simulation is 2d.
    filter_mode : str
        indicates how filtering is done. Possible values are "matrix" or
        "helmholtz". If "matrix", then density/sensitivity filters are
        implemented via a sparse matrix and applied by multiplying
        said matrix with the densities/sensitivities.
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
    obj_func : callable
        objective function. Should update the objective value, the rhs of the
        the adjoint problem (currently only for stationary lin. problems) and 
        a flag indicating whether the objective is self adjoint.
    obj_kw : dict
        keywords needed for the objective function. E. g. for a compliant 
        mechanism and maximization of the displacement it would be the 
        indicator array for output nodes. Check the objective for the necessary
        entries.
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements.
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
    optimizer: str
        solver options which are "oc", "mma" and "gcmma" for the optimality
        criteria method, the method of moving asymptotes and the globally
        covergent method of moving asymptotes.
    optimizer_kw : dict
        dictionary with parameters for optimizer.
    alpha : None or float,
        mixing parameter for design variable update.
    nouteriter: int
        number of TO iterations
    ninneriter: int
        number of inner iterations for GCMMA
    display : bool
        if True, plot design evolution to screen
    file : str
        name of output files
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
        to_log = init_logging(logfile=file)
        #
        to_log(f"minimum compliance problem with optimizer {optimizer}")
        to_log(f"number of spatial dimensions: {ndim}")
        if ndim == 2:
            to_log(f"elements: {nelx} x {nely}")
        elif ndim == 3:
            to_log(f"elements: {nelx} x {nely} x {nelz}")
        if volfrac is not None:
            to_log(f"volfrac: {volfrac} rmin: {rmin}  penal: {penal}")
        else:
            to_log(f"rmin: {rmin}  penal: {penal}")
        to_log("filter: " + ["Sensitivity based",
                             "Density based",
                             "Haeviside Guest",
                             "Haeviside complement Sigmund 2007",
                             "Haeviside eta projection",
                             "Volume Preserving eta projection",
                             "No filter"][ft])
        to_log(f"filter mode: {filter_mode}")
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
    # initialize arrays for gradients
    dobj = np.zeros(x.shape[0],order="F")
    dv = np.ones(x.shape[0],order="F")
    # initialize solver
    if optimizer_kw is None:        
        if optimizer in ["oc","ocm","ocg"]:
            # must be initialized to use the NGuyen/Paulino OC approach
            g = 0
        elif optimizer == "mma":
            # mma needs results of the two previous iterations
            nhistory = 2
            xhist = [x.copy(),x.copy()]
            #
            if optimizer_kw is None:
                optimizer_kw = mma_defaultkws(x.shape[0],ft=ft,n_constr=1)
        elif optimizer == "gcmma":
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
            raise ValueError("Unknown optimizer: ", optimizer)
    # Max and min Young's modulus
    Emin = 1e-9
    Emax = 1.0
    # get element stiffness matrix
    if ndim == 2:
        KE = lk_linear_elast_2d() #lk_poisson_2d()#
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        n_ndof = int(KE.shape[-1]/4)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*n_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat2d(nelx=nelx,nely=nely,
                                                   nnode_dof=n_ndof)
    elif ndim == 3:
        KE = lk_linear_elast_3d()
        # infer nodal degrees of freedom assuming that we have 4/8 nodes in 2/3
        n_ndof = int(KE.shape[-1]/8)
        # number of degrees of freedom
        ndof = (nelx+1)*(nely+1)*(nelz+1)*n_ndof
        # element degree of freedom matrix plus some helper indices
        edofMat, n1, n2, n3, n4 = create_edofMat3d(nelx=nelx,nely=nely,nelz=nelz,
                                                   nnode_dof=n_ndof)
    # Construct the index pointers for the coo format
    iK,jK = create_matrixinds(edofMat,mode=assembly_mode)
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
                                      rmin=rmin,ndim=ndim)
    elif filter_mode == "convolution":
        h,hs = assemble_convolution_filter(nelx=nelx,nely=nely,nelz=nelz,
                                           rmin=rmin,
                                           mapping=mapping,
                                           invmapping=invmapping)
    elif filter_mode == "helmholtz" and ft in [0,1]:
        KF,TF = assemble_helmholtz_filter(nelx=nelx,nely=nely,nelz=nelz,
                                          rmin=rmin,
                                          n1=n1,n2=n2,n3=n3,n4=n4)
        # LU decomposition. returns a function for solving, not the matrices
        lu_solve = factorized(KF)
    # BC's and support
    u,f,fixed,free,springs = bcs(nelx=nelx,nely=nely,nelz=nelz,
                                 ndof=ndof)
    f0 = None
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
            # marching cubes to find contour line
            verts, faces, normals, values = marching_cubes(mapping(-xPhys), 
                                                          level=volfrac)
            fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
            #
            mesh = Poly3DCollection(verts[faces])
            mesh.set_edgecolor('k')
            ax.add_collection3d(mesh)
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
    # optimization loop
    for loop in np.arange(nouteriter):
        # solve FEM, calculate obj. func. and gradients.
        # for
        if optimizer in ["oc","mma", "ocm","ocg"] or\
           (optimizer in ["gcmma"] and ninneriter==0) or\
           loop==0:
            # update physical properties of the elements and thus the entries 
            # of the elements
            if assembly_mode == "full":
                sK = (KE.flatten()[:,None]*(Emin+(xPhys)
                       ** penal*(Emax-Emin))).flatten(order='F')
            # Setup and solve FE problem
            # To Do: loop over boundary conditions if incompatible
            # assemble system matrix
            K = assemble_matrix(sK=sK,iK=iK,jK=jK,
                                ndof=ndof,solver=lin_solver,
                                springs=springs)
            # assemble right hand side
            rhs = assemble_rhs(f0=f,solver=lin_solver)
            # apply boundary conditions to matrix
            K = apply_bc(K=K,solver=lin_solver,
                         free=free,fixed=fixed)
            # solve linear system. fact is a factorization and precond a preconditioner
            u[free, :], fact, precond = solve_lin(K=K, rhs=rhs[free], 
                                                  solver=lin_solver,
                                                  preconditioner=preconditioner)
            # Objective and objective gradient
            obj = 0
            dobj[:] = np.zeros(x.shape[0])
            for i in np.arange(f.shape[1]):
                # obj. value, selfadjoint variables, self adjoint flag
                obj,rhs_adj,self_adj = obj_func(obj=obj, 
                                                xPhys=xPhys,u=u[:,i],
                                                KE=KE,edofMat=edofMat,
                                                Amax=Emax,Amin=Emin,
                                                penal=penal,
                                                **obj_kw)
                # if problem not self adjoint, solve for adjoint variables and
                # calculate derivatives, else use analytical solution
                if self_adj:
                    dobj[:] += rhs_adj
                else:
                    h = np.zeros(f.shape)
                    h[free],_,_ = solve_lin(K, rhs=rhs_adj[free],
                                            solver=lin_solver, P=precond,
                                            preconditioner = preconditioner)
                    if f0 is None:
                        dobj += penal*xPhys**(penal-1)*(Emax-Emin)*\
                              (np.dot(h[edofMat,i], KE)*u[edofMat,i]).sum(1) 
                    else:
                        dobj += penal*xPhys**(penal-1)*(Emax-Emin)*\
                              (np.dot(h[edofMat,i], KE)*\
                               (u[edofMat,i]-f0[:])).sum(1)
                if debug:
                    print("FEM: it.: {0}, problem: {1}, min. u: {2:.10f}, med. u: {3:.10f}, max. u: {4:.10f}".format(
                           loop,i,np.min(u[:,i]),np.median(u[:,i]),np.max(u[:,i])))
        # Constraints and constraint gradients
        if volfrac is not None:
            volconstr = np.array([xPhys.mean() - volfrac])
            if optimizer in ["mma","gcmma"]:
                dv[:] = np.ones(x.shape[0]) /(x.shape[0]*volfrac)
            elif optimizer in ["oc","ocm","ocg"]:
                dv[:] = np.ones(x.shape[0])
        if debug:
            print("Pre-Sensitivity Filter: it.: {0}, dobj: {1:.10f}, dv: {2:.10f}".format(
                   loop,
                   np.max(dobj),
                   np.min(dv)))
        # Sensitivity filtering:
        if ft == 0 and filter_mode == "matrix":
            dobj[:] = np.asarray((H*(x*dobj))[None].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
            #dobj[:] = H @ (dc*x) / Hs / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "convolution":
            dobj[:] = invmapping(convolve(mapping(dobj/hs),
                               h,
                               mode="constant",
                               cval=0)) / np.maximum(0.001, x)
        elif ft == 0 and filter_mode == "helmholtz":
            dobj[:] = TF.T @ lu_solve(TF@(dobj*xPhys))/np.maximum(0.001, x)
        elif ft == 1 and filter_mode == "matrix":
            dobj[:] = np.asarray(H*(dobj[None].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[None].T/Hs))[:, 0]
            #dobj[:] = H @ (dobj/Hs)
            #dv[:] = H @ (dv/Hs)
        elif ft == 1 and filter_mode == "convolution":
            dobj[:] = invmapping(convolve(mapping(dobj/hs),
                                        h,
                                        mode="constant",
                                        cval=0))
            dv[:] = invmapping(convolve(mapping(dv/hs),
                                        h,
                                        mode="constant",
                                        cval=0))
        elif ft == 1 and filter_mode == "helmholtz":
            dobj[:] = TF.T @ lu_solve(TF@dobj)
            dv[:] = TF.T @ lu_solve(TF@dv)
        elif ft == -1:
            pass
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, max. dobj: {1:.10f}, min. dv: {2:.10f}".format(
                   loop,
                   np.max(dobj),
                   np.min(dv)))
        # density update by solver
        xold[:] = x
        # optimality criteria
        if optimizer=="oc":
            (x[:], g) = oc_top88(x=x, volfrac=volfrac, 
                                 dc=dobj, dv=dv, g=g, 
                                 el_flags=el_flags)
        elif optimizer=="ocm":
            (x[:], g) = oc_mechanism(x=x, volfrac=volfrac, 
                                     dc=dobj, dv=dv, g=g, 
                                     el_flags=el_flags)
        elif optimizer=="ocg":
            (x[:], g) = oc_generalized(x=x, volfrac=volfrac, 
                                       dc=dobj, dv=dv, g=g, 
                                       el_flags=el_flags)
        # method of moving asymptotes
        elif optimizer=="mma":
            xval = x.copy()[None].T
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = update_mma(x=x,
                                                                xold1=xhist[-1],
                                                                xold2=xhist[-2],
                                                                xPhys=xPhys,
                                                                obj=obj,
                                                                dobj=dobj,
                                                                constrs=volconstr,
                                                                dconstr=dv,
                                                                iteration=loop,
                                                                **optimizer_kw)
            # update asymptotes 
            optimizer_kw["low"] = low
            optimizer_kw["upp"] = upp
            # delete oldest element of iteration history
            xhist.pop(0)
            xhist.append(xval)
            del xval
            x = xmma.copy().flatten()
        # mixing
        if alpha is not None:
            x = x*(1-alpha) + alpha * xold
        #
        # Anderson acceleration every q steps after q0 iterations
        #if (loop-q0) % q == 0:
        #    # assemble to adequate matrix
        #    xhist = np.column_stack(xhist)
        #    rhist = np.column_stack(rhist)
        #    # differences of x and residuals
        #    dx = xhist[:,1:] - xhist[,:-1]
        #    dr = rhist[:,1:] - rhist[,:-1]
        #    # Solve for optimal update
        #    gamma = lsq_linear(r,np.zeros(n))
        #    # Anderson update
        #    if gamma.success:
        #        x = history_x[-1] - (F[-1] @ gamma).reshape(x.shape)
        #
        # update history
        #else:
        #    xhist.append(x)
        #    rhist.append(x-xold)
        #    # boils effectively down to standard mixing
        #    x += alpha * rhist[-1]
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
            to_log("it.: {0} obj.: {1:.10f} vol.: {2:.10f} ch.: {3:.10f}".format(
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
        export_vtk(filename=file,
                   nelx=nelx,nely=nely,nelz=nelz,
                   xPhys=xPhys,x=x,
                   u=u,f=f,volfrac=volfrac)
    return x, obj
