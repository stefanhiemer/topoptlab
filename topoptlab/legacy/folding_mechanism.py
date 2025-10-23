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

from topoptlab.output_designs import export_vtk,export_stl,threshold
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.fem import update_indices
from topoptlab.optimizer.optimality_criterion import oc_mechanism
from topoptlab.optimizer.mma_utils import update_mma

from mmapy import gcmmasub,asymp,concheck,raaupdate

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         pde=False, passive=False,
         solver="oc", nouteriter=2000, ninneriter=15,
         expansion=0.05,file="folding",
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
        if isfile(".".join([file,"log"])):
            remove(".".join([file,"log"]))   
        # check if any previous loggers exist and close them properly, 
        # otherwise you start writing the same information in a single huge 
        # file
        logger = logging.getLogger()
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()  
        #
        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            handlers=[
                                logging.FileHandler(".".join([file,"log"])),
                                logging.StreamHandler()])
        #
        logging.info(f"Compliant mechanism problem with {solver}")
        logging.info(f"nodes: {nelx} x {nely}")
        logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
        logging.info("Filter method: " + ["Sensitivity based", "Density based",
                                   "Haeviside","No filter"][ft])
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # stiffness constants springs
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
            move = 0.1
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
    if not pde and ft in [0,1]:
        nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        i = np.floor(el/nely)
        j = el%nely
        kk1 = np.maximum(i-(np.ceil(rmin)-1), 0).astype(int)
        kk2 = np.minimum(i+np.ceil(rmin), nelx).astype(int)
        ll1 = np.maximum(j-(np.ceil(rmin)-1), 0).astype(int)
        ll2 = np.minimum(j+np.ceil(rmin), nely).astype(int)
        n_neigh = (kk2-kk1)*(ll2-ll1)
        el,i,j = np.repeat(el, n_neigh),np.repeat(i, n_neigh),np.repeat(j, n_neigh)
        cc = np.arange(el.shape[0])
        k,l = np.hstack([np.stack([a.flatten() for a in \
                         np.meshgrid(np.arange(k1,k2),np.arange(l1,l2))]) \
                         for k1,k2,l1,l2 in zip(kk1,kk2,ll1,ll2)])
        fac = rmin-np.sqrt(((i-k)**2+(j-l)**2))
        iH[cc] = el # row
        jH[cc] = k*nely+l #column
        sH[cc] = np.maximum(0.0, fac)
        # Finalize assembly and convert to csc format
        H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
        Hs = H.sum(1)
    elif pde and ft in [0,1]:
        Rmin = rmin/(2*np.sqrt(3))
        KEF = (Rmin**2) * np.array([[4, -1, -2, -1],
                                    [-1, 4, -1, -2],
                                    [-2, -1, 4, -1],
                                    [-1, -2, -1, 4]])/6 + \
                          np.array([[4, 2, 1, 2],
                                    [2, 4, 2, 1],
                                    [1, 2, 4, 2],
                                    [2, 1, 2, 4]])/36
        ndofF = (nelx+1)*(nely+1)
        edofMatF = np.column_stack((n1, n2, n2 +1, n1 +1 ))
        iKF = np.kron(edofMatF, np.ones((4, 1))).flatten()
        jKF = np.kron(edofMatF, np.ones((1, 4))).flatten()
        sKF = np.tile(KEF.flatten(),nelx*nely)
        KF = coo_matrix((sKF, (iKF, jKF)), shape=(ndofF, ndofF)).tocsc()
        #LF = coo_matrix(cholesky(KF.toarray(),lower=True)).tocsc()
        #LU = splu(KF)
        iTF = edofMatF.flatten(order='F')
        jTF = np.tile(el, 4)
        sTF = np.full(4*nelx*nely,1/4)
        TF = coo_matrix((sTF, (iTF, jTF)), shape=(ndofF,nelx*nely)).tocsc()
    # BC's and support
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], # symmetry 
                       np.array([2*(nely+1)-1])) # bottom support
    dout = ndof - 2 * (nely+1) + 2 
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # Set load
    f0 = KE.dot(expansion/2*np.array([-1.,-1.,1.,-1.,1.,1.,-1.,1.]))
    f[dout,1] = 1
    # general
    free = np.setdiff1d(dofs, fixed)
    # get rid of fixed degrees of freedom from stiffness matrix 
    mask = ~(np.isin(iK,fixed) | np.isin(jK,fixed))
    #
    _dout = update_spring(np.array([dout]), fixed,
                          np.isin(np.array([dout]),fixed))
    iK,jK = update_indices(iK, fixed, mask),update_indices(jK, fixed, mask)
    ndof_free = ndof - fixed.shape[0]
    # passive elements
    if passive == 1:
        el = np.arange(nelx*nely)
        i = np.floor(el/nely)
        j = el%nely
        pass_el = np.sqrt( (j+1-nely/2)**2 + (i+1-nelx/3)**2) < nely/3
    elif passive == 2:
        # top and bottom
        inds = np.hstack((np.arange(0,nelx*nely,nely), # top 
                          np.arange(nely-1,nelx*nely,nely), # bottom
                          np.arange((nelx-1)*nely + 1,nelx*nely-1))) # right side
        pass_el = np.zeros(nelx*nely,dtype=int)
        # set to active
        pass_el[inds] = 2
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
        ax.axis("off")
        fig.show()
    # gradient for the volume constraint is constant regardless of iteration
    # optimization loop
    for loop in np.arange(nouteriter):
        # basic stiffness matrix
        sK = (KE.flatten()[:,None]*(Emin+(xPhys)
              ** penal*(Emax-Emin))).flatten(order='F')[mask]
        K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
        # add springs to stiffness matrix
        #K[_din,_din] += kin # 
        K[_dout,_dout] += kout #
        # set up forces
        f[:,0] = np.zeros(ndof)
        np.add.at(f[:,0],
                  edofMat,
                  (Emin+(xPhys)**penal*(Emax-Emin))[:,None]*f0[None,:])
        # Solve systems
        u[free, :] = spsolve(K, f[free, :])
        # Objective and sensitivity
        obj = -u[dout,0].copy()
        dc = penal*xPhys**(penal-1)*(Emax-Emin)*\
             (np.dot(u[edofMat,1], KE)*(u[edofMat,0]-f0[None,:])).sum(1)
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
        if ft == 0 and not pde:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
        # solve KF y = TF@(dc*xPhys), dc_updated = TF.T @ y
        elif ft == 0 and pde:
            #dc[:] = (TF.T@spsolve_triangular(LF.T, 
            #                (spsolve_triangular(LF,TF@(dc*xPhys))),
            #                lower=False)) \
            #         /np.maximum(0.001, x)
            #dc[:] = TF.T @ LU.solve(TF@(dc*xPhys))/np.maximum(0.001, x)
            dc[:] = TF.T @ spsolve(KF,TF@(dc*xPhys))/np.maximum(0.001, x)
        elif ft == 1 and not pde:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        # solve KF y = TF@dc, dc_updated = TF.T @ y
        # solve KF z = TF@dv, dv_updated = TF.T @ z
        elif ft == 1 and pde:
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
            (x[:], g) = oc_mechanism(x, volfrac, dc, dconstrs[:,0], g, pass_el)
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
        elif ft == 1 and not pde:
            xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
        elif ft == 1 and pde:
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
            logging.info("it.: {0} , obj.: {1:.8f} Vol.: {2:.8f}, ch.: {3:.8f}".format(
            loop, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break 
    # threshold
    xThresh = threshold(xPhys[:,None],volfrac)[:,0]
    #
    # basic stiffness matrix
    sK = (KE.flatten()[:,None]*(Emin+(xThresh)
          ** penal*(Emax-Emin))).flatten(order='F')[mask]
    K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
    # add springs to stiffness matrix
    #K[_din,_din] += kin # 
    #K[_dout,_dout] += kout #
    # set up forces
    f_bw = np.zeros((ndof,1))
    np.add.at(f_bw[:,0],
              edofMat,
              (Emin+(xThresh)**penal*(Emax-Emin))[:,None]*f0[None,:])
    # Solve systems
    u_bw = np.zeros((ndof,1))
    u_bw[free,0] = spsolve(K, f_bw[free])
    obj = -u[dout,0].copy()
    # final output
    logging.info("final: obj.: {0:.8f} Vol.: {1:.8f}".format(obj,xThresh.mean()))
    #
    if display:
        # Plot to screen
        im.set_array(-xThresh.reshape((nelx, nely)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        #
        plt.show()
        input("Press any key...")
    if export:
        #
        export_vtk(filename=file, 
                   nelx=nelx,nely=nely, 
                   xPhys=xPhys[:,None],x=x, 
                   u=u,f=f,
                   u_bw=u_bw,f_bw=f_bw,
                   volfrac=volfrac)
        export_stl(filename=file, 
                   nelx=nelx,nely=nely, 
                   xPhys=xPhys[:,None],
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