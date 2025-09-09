# SPDX-License-Identifier: GPL-3.0-or-later
# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
from os.path import isfile
from os import remove
import logging 
import traceback

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from topoptlab.output_designs import export_vtk
from topoptlab.filters import find_eta
from topoptlab.optimality_criterion import oc_haevi
from topoptlab.fem import update_indices
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d

from mmapy import mmasub,gcmmasub,asymp,concheck,raaupdate
    
projections = [2,3,4,5]
filters = [0,1]
# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         pde=False, passive=False, solver="oc", 
         nouteriter=2000, ninneriter=15,
         verbose=True,
         debug=False):
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

    Returns
    -------
    None.

    """
    # check if log file exists and if true delete
    if isfile("topopt_haevi.log"):
        remove("topopt_haevi.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler("topopt_haevi.log"),
                        logging.StreamHandler()])
    #
    logging.info(f"Minimum compliance problem with {solver}")
    logging.info(f"nodes: {nelx} x {nely}")
    logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
    logging.info("Filter method: " + ["Sensitivity based", 
                               "Density based",
                               "Haeviside Guest",
                               "Haeviside complement Sigmund 2007",
                               "Haeviside eta projection",
                               "Volume Preserving eta projection",
                               "No filter"][ft])
    # Max and min Young's modulus
    Emin = 1e-9
    Emax = 1.0
    # dofs:
    ndof = 2*(nelx+1)*(nely+1)
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely*nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    # these are Heavisde projections
    if ft in projections:
        beta = 1
        xTilde = x.copy()
        if solver in ["mma","gcmma"] and ft in [2,3,4]:
            raise NotImplementedError("This combination is currently not implemented.")
    if ft in [2]:
        eta=None
        xPhys = 1 - np.exp(-beta*xTilde) + xTilde*np.exp(-beta)
    elif ft in [3]:
        eta = None
        xPhys = np.exp(-beta*(1-xTilde)) - (1-xTilde)*np.exp(-beta)
    elif ft in [4,5]:
        if ft in [4]: 
            eta = 0.5
        elif ft in [5]:
            eta = find_eta(0.5, xTilde, beta, volfrac) 
        xPhys = (np.tanh(beta*eta)+np.tanh(beta*(xTilde - eta)))/\
                (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
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
        if ft in [5]: 
            move = 0.1
        else:
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
        if ft in [5]: 
            move = 0.1
        else:
            move = 0.2
        #
        epsimin = 0.0000001
        raa0 = 0.01
        raa = 0.01*np.ones((m,1))
        raa0eps = 0.000001
        raaeps = 0.000001*np.ones((m,1))
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
    if not pde:
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
    elif pde:
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
    elif pde and ft in [2]:
        raise ValueError("PDE filter and Heaviside projection incompatible.")
    # BC's and support
    dofs = np.arange(ndof)
    # MBB beam
    fixed = np.union1d(dofs[0:2*(nely+1):2], 
                       np.array([2*(nelx+1)*(nely+1)-1]))
    # cases 5.1 and 5.2
    #fixed = dofs[:2*nely+1]
    # general
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # Set load
    f[1, 0] = -1 # MBB
    #f[2*(nelx+1)*(nely+1)-1,0] = -1 # case 5.1 and 5.2
    #f[2*nelx*(nely+1)+1,1] = 1 # case 5.2
    # get rid of fixed degrees of freedom from stiffness matrix 
    mask = ~(np.isin(iK,fixed) | np.isin(jK,fixed))
    #
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
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots(1,1,figsize=(12,4))
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
    if debug:
        print("x: ", x)
        print("xPhys: ", xPhys)
        if ft == 2:
            print("xTilde: ", xTilde)
        
    loopbeta = 0
    if debug:
        if ft == 2:
            print("it.: {0} , x.: {1:.10f}, xTilde: {2:.10f}, xPhys: {3:.10f},".format(
                   0, np.median(x),np.median(xTilde),np.median(xPhys)), 
                   "g: {0:.10f}".format(g))
        else:
            print("it.: {0} , x.: {1:.10f}, xPhys: {2:.10f},".format(
                   0, np.median(x),np.median(xPhys)), 
                   "g: {0:.10f}".format(g))
    # initialize gradients
    dc = np.zeros(nelx*nely)
    dv = np.ones(nelx*nely)
    # optimization loop
    for loop in np.arange(nouteriter):
        #
        loopbeta += 1 
        # solve FEM, calculate obj. func. and gradients.
        # for 
        if solver in ["oc","mma"] or\
           (solver in ["gcmma"] and ninneriter==0) or\
           loop==0:
            
            # Setup and solve FE problem
            sK = (KE.flatten()[:,None]*(Emin+(xPhys)
                  ** penal*(Emax-Emin))).flatten(order='F')[mask]
            K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
            # Remove constrained dofs from matrix
            #K = K[free, :][:, free]
            # Solve system(s)
            if u.shape[1] == 1:
                u[free, 0] = spsolve(K, f[free, 0])
            else:
                u[free, :] = spsolve(K, f[free, :])
            # Objective and sensitivity
            obj = 0
            dc[:] = np.zeros(nelx*nely)
            for i in np.arange(f.shape[1]):
                #ce = (np.dot(u[edofMat,i].reshape(nelx*nely, KE.shape[0]), KE)
                #         * u[edofMat,i].reshape(nelx*nely, KE.shape[0])).sum(1)
                ui = u[:,i]
                ce = (np.dot(ui[edofMat], KE) * ui[edofMat]).sum(1)
                obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
                dc[:] -= penal*xPhys**(penal-1)*(Emax-Emin)*ce
        dv[:] = np.ones(nelx*nely) 
        if debug:
            print("Pre-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dv: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dv)))
        # Sensitivity filtering:
        if ft in [0] and not pde:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
        elif ft in [0] and pde:
            dc[:] = TF.T @ spsolve(KF,TF@(dc*xPhys))/np.maximum(0.001, x)
        elif ft in [1] and not pde:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0] 
        elif ft in [1] and pde:
            dc[:] = TF.T @ spsolve(KF,TF@dc)
            dv[:] = TF.T @ spsolve(KF,TF@dv)
        elif ft in projections:
            if ft in [2]:
                dx = beta*np.exp(-beta*xTilde) + np.exp(-beta)
            elif ft in [3]:
                dx = np.exp(-beta*(1-xTilde)) * beta \
                     + np.exp(-beta)
            elif ft in [4,5]:
                dx = beta * (1 - np.tanh(beta * (xTilde - eta))**2) /\
                        (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
            dc[:] = np.asarray(H*((dc*dx)[np.newaxis].T/Hs))[:, 0]
            dv[:] = np.asarray(H*((dv*dx)[np.newaxis].T/Hs))[:, 0]
            # safety for division by 0
        elif ft == -1:
            pass
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dv: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dv)))
        # density update by solver
        xold[:] = x
        # optimality criteria
        if solver=="oc" and ft in filters:
            (x[:], g) = oc_haevi(nelx, nely, x, volfrac, dc, dv, g, 
                           pass_el,
                           None,None,None,None,ft,
                           debug)
        elif solver=="oc" and ft in projections:
            (x[:],xTilde[:],xPhys[:],g) = oc_haevi(nelx, nely, x, volfrac, dc, dv, g, 
                                                   pass_el,
                                                   H,Hs,beta,eta,ft,
                                                   debug=debug) 
        # method of moving asymptotes, implementation by Arjen Deetman
        elif solver=="mma":
            xval = x.copy()[np.newaxis].T
            xmma,ymma,zmma,lam,xsi,eta_mma,mu,zet,s,low,upp = update_mma(x,xold1,xold2,xPhys,
                                                                         obj,dc,dv,loop,
                                                                         m,xmin,xmax,
                                                                         low,upp,
                                                                         a0,a,c,d,move)
            xold2 = xold1.copy()
            xold1 = xval.copy()
            x = xmma.copy().flatten()
        # globally convergent method of moving asymptotes, implementation by 
        # Arjen Deetman
        elif solver=="gcmma":
            mu0 = 1.0 # Scale factor for objective function
            mu1 = 1.0 # Scale factor for volume constraint function
            f0val = mu0*obj 
            df0dx = mu0*dc[np.newaxis].T
            fval = mu1*np.array([[xPhys.sum()/x.shape[0]-volfrac]])
            dfdx = mu1*(dv/(x.shape[0]*volfrac))[np.newaxis]
            xval = x.copy()[np.newaxis].T
            #
            low,upp,raa0,raa= \
                  asymp(loop,x.shape[0],xval,xold1,xold2,xmin,xmax,low,upp,
                        raa0,raa,raa0eps,raaeps,df0dx,dfdx)
            xmma,ymma,zmma,lam,xsi,eta_gcmma,mu,zet,s,f0app,fapp= \
                  gcmmasub(m,x.shape[0],loop,epsimin,xval,xmin,xmax,low,upp,
                           raa0,raa,f0val,df0dx,fval,dfdx,a0,a,c,d)
            # calculate objective- and constraint function values
            # constraint functions at point xmma ( optimal solution of 
            # the subproblem).
            # Filter design variables
            if ft == 0:
                xPhys[:] = xmma.copy().flatten()
            elif ft == 1 and not pde:
                xPhys[:] = np.asarray(H*xmma.copy().flatten()[np.newaxis].T/Hs)[:, 0]
            elif ft == 1 and pde:
                xPhys[:] = TF.T @ spsolve(KF,TF@xmma.copy().flatten())
            elif ft == -1:
                xPhys[:]  = xmma.copy().flatten()
            # Setup and solve FE problem
            sK = (KE.flatten()[:,None]*(Emin+(xPhys)
                  ** penal*(Emax-Emin))).flatten(order='F')[mask]
            K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
            # Remove constrained dofs from matrix
            #K = K[free, :][:, free]
            # Solve system(s)
            if u.shape[1] == 1:
                u[free, 0] = spsolve(K, f[free, 0])
            else:
                u[free, :] = spsolve(K, f[free, :])
            # Objective and sensitivity
            obj = 0
            for i in np.arange(f.shape[1]):
                #ce = (np.dot(u[edofMat,i].reshape(nelx*nely, KE.shape[0]), KE)
                #         * u[edofMat,i].reshape(nelx*nely, KE.shape[0])).sum(1)
                ui = u[:,i]
                ce = (np.dot(ui[edofMat].reshape(nelx*nely, KE.shape[0]), KE)
                         * ui[edofMat].reshape(nelx*nely, KE.shape[0])).sum(1)
                obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
            # constraint
            constr = xPhys.mean() - volfrac
            # check if approximations conservative
            conserv = concheck(m,epsimin,f0app,obj,fapp,constr)
            # inner iterations
            if conserv == 0:
                for innerit in np.arange(ninneriter):
                    # update raa0 and raa
                    raa0,raa = raaupdate(xmma,xval,xmin,xmax,low,upp,
                                         obj,constr,f0app,fapp,raa0, 
                                         raa,raa0eps,raaeps,epsimin)
                    # solve subproblem with new raa0 and raa:
                    xmma,ymma,zmma,lam,xsi,eta_gcmma,mu,zet,s,f0app,fapp = gcmmasub(m,
                                                    x.shape[0],loop,epsimin,xval,xmin, 
                                                    xmax,low,upp,raa0,raa,f0val,
                                                    df0dx,fval,dfdx,a0,a,c,d)
                    # calculate objective- and constraint function values
                    # constraint functions at point xmma ( optimal solution of 
                    # the subproblem).
                    # Filter design variables
                    if ft == 0:
                        xPhys[:] = xmma.copy().flatten()
                    elif ft == 1 and not pde:
                        xPhys[:] = np.asarray(H*xmma.copy().flatten()[np.newaxis].T/Hs)[:, 0]
                    elif ft == 1 and pde:
                        xPhys[:] = TF.T @ spsolve(KF,TF@xmma.copy().flatten())
                    elif ft == -1:
                        xPhys[:]  = xmma.copy().flatten()
                    # Setup and solve FE problem
                    sK = (KE.flatten()[:,None]*(Emin+(xPhys)
                          ** penal*(Emax-Emin))).flatten(order='F')[mask]
                    K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
                    # Remove constrained dofs from matrix
                    #K = K[free, :][:, free]
                    # Solve system(s)
                    if u.shape[1] == 1:
                        u[free, 0] = spsolve(K, f[free, 0])
                    else:
                        u[free, :] = spsolve(K, f[free, :])
                    # objective function
                    obj = 0
                    for i in np.arange(f.shape[1]):
                        #ce = (np.dot(u[edofMat,i].reshape(nelx*nely, KE.shape[0]), KE)
                        #         * u[edofMat,i].reshape(nelx*nely, KE.shape[0])).sum(1)
                        ui = u[:,i]
                        ce = (np.dot(ui[edofMat].reshape(nelx*nely, KE.shape[0]), KE)
                                 * ui[edofMat].reshape(nelx*nely, KE.shape[0])).sum(1)
                        obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
                    # constraint
                    constr = xPhys.mean() - volfrac
                    # It is checked if the approximations have become conservative:
                    conserv = concheck(m,epsimin,f0app,obj,fapp,constr)
                    if conserv:
                        break
            # calculate gradients/sensitivity
            dc = 0
            for i in np.arange(f.shape[1]):
                dc -= penal*xPhys**(penal-1)*(Emax-Emin)*ce
            #
            xold2 = xold1.copy()
            xold1 = xval.copy()
            x = xmma.copy().flatten()
        if debug:
            print("Post Density Update: it.: {0}, dc: {1:.10f}, dv: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dv)))
            print("Post-Density Update: it.: {0} , x.: {1:.10f}, xPhys: {2:.10f},".format(
                   loop+1, np.median(x),np.median(xPhys)), 
                   "g: {0:.10f}".format(g))
        # Filter design variables
        if ft in [0,-1]:
            xPhys[:] = x
        elif ft in [1] and not pde:
            xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
        elif ft in [1] and pde:
            xPhys[:] = TF.T @ spsolve(KF,TF@x)
        elif ft in [5]:
            xTilde = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
            eta = find_eta(eta, xTilde, beta, volfrac)
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xTilde - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        # Compute the change by the inf. norm
        change = (np.abs(x-xold)).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if debug == 1:
            if ft in projections:
                print("Post-Density Filter: it.: {0} , x.: {1:.10f}, xTilde: {2:.10f}, xPhys: {3:.10f},".format(
                       loop+1, np.median(x),np.median(xTilde),np.median(xPhys)), 
                       "g: {0:.10f}".format(g))
            else:
                print("Post-Density Filter: it.: {0} , x.: {1:.10f}, xPhys: {2:.10f},".format(
                       loop+1, np.median(x),np.median(xPhys)), 
                       "g: {0:.10f}".format(g))
        if verbose: 
            logging.info("it.: {0} , obj.: {1:.10f}, Vol.: {2:.10f}, ch.: {3:.10f}".format(
            loop+1, obj, xPhys.mean(), change))
        # convergence check and continuation
        if change < 0.01 and ft in filters:
            break
        elif (ft in projections) and (beta < 512) and \
            (loopbeta >= 50 or change < 0.01):
            beta = 2 * beta
            loopbeta = 0
            logging.info(f"Parameter beta increased to {beta}")
        elif (ft in projections) and (beta >= 512) and (change < 0.01):
            break
    # 
    plt.show()
    logging.shutdown()
    input("Press any key...")
    #
    if ft in [0,1]:
        xTilde=None
    export_vtk(filename="topopt_haevi", 
               nelx=nelx,nely=nely, 
               xPhys=xPhys,x=x, xTilde=xTilde,
               u=u,f=f,volfrac=volfrac)
    return x, obj

def update_mma(x,xold1,xold2,xPhys,obj,dc,dv,iteration,
               m,xmin,xmax,low,upp,a0,a,c,d,move):
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj 
    df0dx = mu0*dc[np.newaxis].T
    fval = mu1*np.array([[xPhys.sum()/x.shape[0]-volfrac]])
    dfdx = mu1*(dv/(x.shape[0]*volfrac))[np.newaxis]
    xval = x.copy()[np.newaxis].T 
        
    return mmasub(m,x.shape[0],iteration,xval,xmin,xmax,
                  xold1,xold2,f0val,df0dx,
                  fval,dfdx,low,upp,a0,a,c,d,move)

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60  # 180
    nely = int(nelx/3)  # 60
    volfrac = 0.5  # 0.4
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 1 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv) > 1:
        nelx = int(sys.argv[1])
    if len(sys.argv) > 2:
        nely = int(sys.argv[2])
    if len(sys.argv) > 3:
        volfrac = float(sys.argv[3])
    if len(sys.argv) > 4:
        rmin = float(sys.argv[4])
    if len(sys.argv) > 5:
        penal = float(sys.argv[5])
    if len(sys.argv) > 6:
        ft = int(sys.argv[6])
    try:
        main(nelx, nely, volfrac, penal, rmin, ft, 
             passive=False,pde=False,solver="mma",
             nouteriter=2000,
             ninneriter=0,
             debug=False)
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.shutdown()
