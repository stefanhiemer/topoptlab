# SPDX-License-Identifier: GPL-3.0-or-later
# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
from os.path import isfile
from os import remove
import logging 

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve,spsolve_triangular,splu,cg
from scipy.linalg import cholesky
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from topoptlab.output_designs import export_vtk,export_stl
from topoptlab.fem import update_indices
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.optimizer.optimality_criterion import oc_mechanism
from topoptlab.optimizer.mma_utils import update_mma
from topoptlab.filters import assemble_matrix_filter,assemble_helmholtz_filter

from mmapy import gcmmasub,asymp,concheck,raaupdate

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         filter_mode=False, passive=False,
         solver="oc", nouteriter=2000, ninneriter=15,
         display=True, export=True, write_log=True):
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
    pde: boolean
        if true, Helmholtz filter is used
    passive: boolean
        if true, passive elements to form a cricle are created.
    solver: str
        solver options which are "oc", "mma" and "gcmma" for the optimality 
        criteria method, the method of moving asymptotes and the globally 
        covergent method of moving asymptotes.
    nouteriter: int 
        number of total TO iterations
    ninneriter: int
        number of inner iterations for GCMMA
    display: bool
        if True, plot design evolution to screen
    export: bool
        if True, export design as vtk file.
    write_log: bool
        if True, write a log file and display results to command line.
    Returns
    -------
    None.

    """
    if write_log:
        # check if log file exists and if true delete
        if isfile("topoptm.log"):
            remove("topoptm.log")
        logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler("topoptm.log"),
                            logging.StreamHandler()])
        #
        logging.info("Compliant mechanism problem with OC")
        logging.info(f"nodes: {nelx} x {nely}")
        logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
        logging.info("Filter method: " + ["Sensitivity based", "Density based",
                                   "Haeviside","No filter"][ft])
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # stiffness constants springs 
    kin = 0.1
    kout = 0.1
    # dofs:
    ndof = 2*(nelx+1)*(nely+1)
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely*nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    # initialize solver
    if solver=="oc":
        # must be initialized to use the NGuyen/Paulino OC approach
        g = 0
        n_constr = 1
    elif solver == "mma":
        # number of constraints.
        n_constr = 0
        if volfrac is not None:
            n_constr += 1 
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
        a = np.zeros((n_constr,1)) 
        c = 10000*np.ones((n_constr,1))
        d = np.zeros((n_constr,1))
        if ft in [5,6]: 
            move = 0.1
        elif ft in [7]:
            move = 0.05
        else:
            move = 0.2
    elif solver == "gcmma":
        # number of constraints.
        n_constr = 0
        if volfrac is not None:
            n_constr += 1 
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
        a = np.zeros((n_constr,1)) 
        c = 10000*np.ones((n_constr,1))
        d = np.zeros((n_constr,1))
        if ft in [5,6]: 
            move = 0.1
        elif ft in [7]:
            move = 0.05
        else:
            move = 0.2
        #
        epsimin = 0.0000001
        raa0 = 0.01
        raa = 0.01*np.ones((n_constr,1))
        raa0eps = 0.000001
        raaeps = 0.000001*np.ones((n_constr,1))
    else:
        raise ValueError("Unknown solver: ", solver)
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk_linear_elast_2d()
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 
                               2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    if filter_mode == "matrix" and ft in [0,1]:
        H,Hs = assemble_matrix_filter(nelx=nelx,nely=nely,rmin=rmin)
    elif filter_mode == "helmholtz" and ft in [0,1]:
        KF,TF = assemble_helmholtz_filter(nelx=nelx,nely=nely,rmin=rmin,
                                          n1=n1,n2=n2)
    # BC's and support
    dofs = np.arange(ndof)
    fixed = np.union1d(np.arange(1,(nelx+1)*(nely+1)*2,(nely+1)*2), # symmetry
                       np.arange(2*(nely+1)-4,2*(nely+1))) # bottom left bit
    din = 0
    dout = 2*nelx*(nely+1)
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # Set load
    f[din,0] = 1
    f[dout,1] = -1
    # general
    free = np.setdiff1d(dofs, fixed)
    # get rid of fixed degrees of freedom from stiffness matrix 
    mask = ~(np.isin(iK,fixed) | np.isin(jK,fixed))
    #
    _din, _dout = update_spring(np.array([din,dout]), fixed, 
                                np.isin(np.array([din,dout]),fixed))
    iK,jK = update_indices(iK, fixed, mask),update_indices(jK, fixed, mask)
    ndof_free = ndof - fixed.shape[0]
    # passive elements
    if passive:
        el = np.arange(nelx*nely)
        i = np.floor(el/nely)
        j = el%nely
        pass_el = np.sqrt( (j+1-nely/2)**2 + (i+1-nelx/3)**2) < nely/3
    else:
        pass_el = None
    if display:
        # Initialize plot and plot the initial design
        plt.ion()  # Ensure that redrawing is possible
        fig, ax = plt.subplots(1,1)
        im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
                       interpolation='none', norm=Normalize(vmin=-1, vmax=0))
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        fig.show()
    # initialize gradients
    dc = np.zeros(nelx*nely)
    # optimization loop
    for loop in np.arange(nouteriter):
        # Setup and solve FE problem
        sK = (KE.flatten()[:,None]*(Emin+(xPhys)
              ** penal*(Emax-Emin))).flatten(order='F')[mask]
        K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
        # add springs to stiffness matrix
        K[_din,_din] += kin # 
        K[_dout,_dout] += kout # 
        # Solve systems
        u[free, 0] = spsolve(K, f[free,0])# :])
        # Objective and sensitivity
        obj = 0
        dc[:] = np.zeros(nelx*nely)
        obj,dc[:] = var_maximization(xPhys=xPhys, u=u[:,0],
                                     l=f[:,1], free=free, inds_out=np.array([dout]),
                                     K=K, KE=KE, edofMat=edofMat,
                                     Amax=Emax, Amin=Emin, penal=penal,
                                     obj=obj,dc=dc,f0=None)
        #obj = u[dout,0].copy()
        #dc = penal*xPhys**(penal-1)*(Emax-Emin)*(np.dot(u[edofMat,1], KE)*\
        #      u[edofMat,0]).sum(1)
        # constraints and derivatives/sensitivities of constraints
        constrs = []
        dconstrs = []
        # derivative of volume constraint
        if volfrac is not None:
            constrs.append(xPhys.mean() - volfrac)
            if solver in ["mma","gcmma"]:
                dconstrs.append(np.ones(nely*nelx)/(x.shape[0]*volfrac))
            elif solver in ["oc"]:
                dconstrs.append(np.ones(nely*nelx))
        # merge to np.array. squeeze is only there for consistency
        constrs = np.hstack(constrs)
        dconstrs = np.column_stack(dconstrs)
        # Sensitivity filtering:
        if ft == 0 and filter_mode == "matrix":
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
        # solve KF y = TF@(dc*xPhys), dc_updated = TF.T @ y
        elif ft == 0 and filter_mode == "helmholtz":
            #dc[:] = (TF.T@spsolve_triangular(LF.T, 
            #                (spsolve_triangular(LF,TF@(dc*xPhys))),
            #                lower=False)) \
            #         /np.maximum(0.001, x)
            #dc[:] = TF.T @ LU.solve(TF@(dc*xPhys))/np.maximum(0.001, x)
            dc[:] = TF.T @ spsolve(KF,TF@(dc*xPhys))/np.maximum(0.001, x)
        elif ft == 1 and filter_mode == "matrix":
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        # solve KF y = TF@dc, dc_updated = TF.T @ y
        # solve KF z = TF@dv, dv_updated = TF.T @ z
        elif ft == 1 and filter_mode == "helmholtz":
            #dc[:] = TF.T @ spsolve_triangular(LF.T, 
            #                                  (spsolve_triangular(LF, TF@dc)),
            #                                  lower=False)
            #dv[:] = TF.T @ spsolve_triangular(LF.T, 
            #                                  (spsolve_triangular(LF, TF@dv)),
            #                                  lower=False)
            #dc[:] = TF.T @ LU.solve(TF@dc)
            #dv[:] = TF.T @ LU.solve(TF@dv)
            dc[:] = TF.T @ spsolve(KF,TF@dc)
            dconstrs[:,:] = (TF.T @ spsolve(KF,TF@dconstrs)).reshape(dconstrs.shape)
        elif ft == -1:
            pass
        # Optimality criteria
        if solver=="oc":
            xold[:] = x 
            (x[:], g) = oc_mechanism(x, volfrac, dc, dconstrs[:,0], g, 
                                     el_flags=pass_el)
        # method of moving asymptotes, implementation by Arjen Deetman
        elif solver=="mma":
            xval = x.copy()[np.newaxis].T
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = update_mma(x,xold1,xold2,xPhys,
                                                                         obj,dc,constrs,dconstrs,loop,
                                                                         n_constr,xmin,xmax,
                                                                         low,upp,
                                                                         a0,a,c,d,move)
            xold2 = xold1.copy()
            xold1 = xval.copy()
            x = xmma.copy().flatten()
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1 and filter_mode == "matrix":
            xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
        elif ft == 1 and filter_mode == "helmholtz":
            #xPhys[:] = TF.T @ spsolve_triangular(LF.T, 
            #                                     (spsolve_triangular(LF, TF@x)),
            #                                     lower=False)
            #xPhys[:] = TF.T @ LU.solve(TF@x)
            xPhys[:] = TF.T @ spsolve(KF,TF@x)
        elif ft == -1:
            xPhys[:] = x
            pass
        # Compute the change by the inf. norm
        change = (np.abs(x-xold)).max()
        if display:
            # Plot to screen
            im.set_array(-xPhys.reshape((nelx, nely)).T)
            fig.canvas.draw()
            plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if write_log: 
            logging.info("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(
            loop, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break 
    if display:
        #
        plt.show()
        input("Press any key...")
    if export:
        #
        export_vtk(filename="topoptm", 
                   nelx=nelx,nely=nely, 
                   xPhys=xPhys,x=x, 
                   u=u,f=f,volfrac=volfrac)
        export_stl(filename="topoptm", 
                   nelx=nelx,nely=nely, 
                   xPhys=xPhys,
                   volfrac=volfrac)
    return x, obj

def update_spring(inds,fixed,mask):
    """
    Update the indices for the springs for the in and output.

    Parameters
    ----------
    inds : np.array
        indices of the spring degrees of freedom in the stiffness matrix.
    fixed : np.array
        indices of fixed degrees of freedom.
    mask : np.array
        mask to kick out fixed degrees of freedom.

    Returns
    -------
    inds : np.arrays
        updated indices.

    """
    inds = inds - np.bincount(np.digitize(fixed, inds))[:inds.shape[0]].cumsum()
    
    return inds

def var_maximization(xPhys,u,l,free,inds_out,
                     K,KE,edofMat,
                     Amax,Amin,penal,
                     obj,dc,f0=None,**kwargs):
    """
    Update objective and gradient for maximization of state variable in 
    specified points. The mechanic version of this is the compliant mechanism 
    with maximized displacement.

    Parameters
    ----------
    xPhys : np.ndarray
        SIMP densities of shape (nel).
    u : np.ndarray
        state variable (displacement, temperature) of shape (ndof).
    l : np.ndarray
        indicator vector for state variable of shape (ndof). Is 1 at output 
        nodes.
    free : np.ndarray
        indices of free nodes.
    inds_out : np.ndarray
        indices of nodes where the displacement is to be maximized. shape (nout)
    K : scipy.sparse matrix/array
        global stiffness matrix of shape (ndof).
    KE : np.ndarray
        element stiffness matrix of shape (nedof).
    edofMat : np.ndarray
        element degree of freedom matrix of shape (nel,nedof)
    Amax : float
        maximum value for material property A
    Amin : float
        minimum value for material property A. Should be small compared to Amax
        but not zero and Amax + Amin should recover the property A of the material
    penal: float
        penalty exponent for the SIMP method.
    obj : float
        objective function.
    dc : np.ndarray
        sensitivities/gradients of design variables with regards to objective 
        function. shape (ndesign)
    f0: np.ndarray
        if system is subjected to an affine expansion causing the loads, 
        this is the resulting load for an element of density one. shape (nedof)

    Returns
    -------
    obj : float
        updated objective function.
    dc : np.ndarray
        updated sensitivities/gradients of design variables with regards to 
        objective function. shape (ndesign)

    """
    obj += u[inds_out].sum()
    # solve adjoint problem
    h = np.zeros(l.shape)
    h[free] = spsolve(K, l[free])
    #
    if f0 is None:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h[edofMat], KE)*\
                 u[edofMat]).sum(1)
    else:
        dc[:] += penal*xPhys**(penal-1)*(Amax-Amin)*(np.dot(h, KE)*\
                 (u[edofMat,0]-f0[None,:])).sum(1)
    return obj, dc
