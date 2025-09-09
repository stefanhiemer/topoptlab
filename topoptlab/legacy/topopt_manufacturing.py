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
from topoptlab.filters import AMfilter,find_eta
from topoptlab.fem import update_indices
from topoptlab.elements.linear_elasticity_2d import lk_linear_elast_2d
from topoptlab.optimizer.mma_utils import update_mma

from mmapy import gcmmasub,asymp,concheck,raaupdate
    
projections = [2,3,4,5]
filters = [0,1]
# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft,
         manufact = None, q = 10, baseplate="S",
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
    if isfile("topopt_manufact.log"):
        remove("topopt_manufact.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler("topopt_haevi.log"),
                        logging.StreamHandler()])
    #
    logging.info(f"Minimum compliance problem with {solver} and manufacturing constraints.")
    logging.info(f"nodes: {nelx} x {nely}")
    logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
    logging.info("Filter method: " + ["Sensitivity based", 
                               "Density based",
                               "Haeviside Guest",
                               "Haeviside complement Sigmund 2007",
                               "Haeviside eta projection",
                               "Volume Preserving eta projection",
                               "Additive Manufacturing filter by Langelaar combined with density filter",
                               "No filter"][ft])
    #
    if manufact is not None and solver == "oc":
        raise ValueError("Current Optimality criterion method can only handle volume constraint")
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
    # Haeviside projection Guest 2004
    if ft in [2]:
        eta=None
        xPhys = 1 - np.exp(-beta*xTilde) + xTilde*np.exp(-beta)
    # Haeviside counterpart projection Sigmund 2007
    elif ft in [3]:
        eta = None
        xPhys = np.exp(-beta*(1-xTilde)) - (1-xTilde)*np.exp(-beta)
    # eta projection and volume conserving adaptation in similar style to Xu 2010
    elif ft in [4,5]:
        if ft in [4]: 
            eta = 0.5
        elif ft in [5]:
            eta = find_eta(0.5, xTilde, beta, volfrac)
        xPhys = (np.tanh(beta*eta)+np.tanh(beta*(xTilde - eta)))/\
                (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
    # additive manufacturing filter by Langelaar
    elif ft in [6]:
        beta = None
        eta = None
        xTilde = x.copy()
        # filter needs the densities assembled to rectangular domain
        xPhys = AMfilter(xTilde.reshape((nelx, nely)).T, baseplate)
        xPhys = xPhys.T.flatten()
    elif ft in [7]:
        beta = 1
        eta = 0.5
        xTilde = x.copy()
        xPrint = x.copy()
        # filter needs the densities assembled to rectangular domain
        xPhys = AMfilter(xTilde.reshape((nelx, nely)).T, baseplate)
        xPhys = xPhys.T.flatten()
    # initialize solver
    if solver=="oc":
        # must be initialized to use the NGuyen/Paulino OC approach
        g = 0
        n_constr = 1
    elif solver == "mma":
        # number of constraints.
        n_constr = 0
        if manufact==0:
            n_constr += (nely-1)*nelx
        elif manufact==1:
            n_constr += nely*(nelx-1)
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
        if manufact==0:
            n_constr += (nely-1)*nelx
        elif manufact==1:
            n_constr += nely*(nelx-1)
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
    if not pde and ft in [0,1,2,3,4,5,6,7]:
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
    elif pde and ft in [2,3,4,5]:
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
    # create indices for casting constraints
    if manufact==0:
        i_cast = np.delete(np.arange(nelx*nely),
                           np.arange(-1,nelx*nely,nely))
        j_cast = i_cast+1
    elif manufact==1:
        i_cast = np.arange((nelx-1)*nely)
        j_cast = np.arange(nely,nelx*nely)
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
    # optimization loop
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
            # Objective and sensitivity of objective function
            obj = 0
            dc = 0
            for i in np.arange(f.shape[1]):
                #ce = (np.dot(u[edofMat,i].reshape(nelx*nely, KE.shape[0]), KE)
                #         * u[edofMat,i].reshape(nelx*nely, KE.shape[0])).sum(1)
                ui = u[:,i]
                ce = (np.dot(ui[edofMat], KE) * ui[edofMat]).sum(1)
                obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
                dc -= penal*xPhys**(penal-1)*(Emax-Emin)*ce
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
        # casting constraint for filling from top of y direction
        if manufact in [0]: 
            constrs.append(xPhys[i_cast] - xPhys[j_cast]) 
            dcast = np.zeros((nelx*nely,(nely-1)*nelx))
            dcast[i_cast, np.arange((nely-1)*nelx)] = 1
            dcast[j_cast, np.arange((nely-1)*nelx)] = -1
            dconstrs.append(dcast)
        elif manufact in [1]: 
            constrs.append(xPhys[i_cast] - xPhys[j_cast]) 
            dcast = np.zeros((nelx*nely,(nelx-1)*nely))
            dcast[i_cast, np.arange((nelx-1)*nely)] = -1
            dcast[j_cast, np.arange((nelx-1)*nely)] = 1
            dconstrs.append(dcast)
        # merge to np.array. squeeze is only there for consistency
        constrs = np.hstack(constrs)
        dconstrs = np.column_stack(dconstrs)
        # merge to 
        if debug:
            print("Pre-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dconstrs: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dconstrs)))
        # Sensitivity filtering:
        if ft in [0] and not pde:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
        elif ft in [0] and pde:
            dc[:] = TF.T @ spsolve(KF,TF@(dc*xPhys))/np.maximum(0.001, x)
        elif ft in [1] and not pde:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        elif ft in [1] and pde:
            dc[:] = TF.T @ spsolve(KF,TF@dc)
            dconstrs[:] = TF.T @ spsolve(KF,TF@dconstrs)
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
            dconstrs[:] = np.asarray(H*((dconstrs*dx[:,None])/Hs))
        elif ft in [6]:
            # apply AM filter to sensitivities
            dc[:] = AMfilter(xTilde.reshape((nelx, nely)).T,
                            baseplate,
                            dc.reshape(nelx,nely).T[:,:,None]).T.flatten()
            dconstrs = AMfilter(xTilde.reshape((nelx, nely)).T,
                            baseplate,
                            np.transpose(dconstrs.reshape(nelx,nely,n_constr),
                                         (1,0,2)))
            dconstrs = np.transpose(dconstrs,(1,0,2)).reshape(nelx*nely,n_constr)
            if debug:
                print("Intermediate-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dconstrs: {2:.10f}".format(
                       loop, 
                       np.max(dc),
                       np.min(dconstrs)))
            # apply density filter to sensitivities
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        elif ft in [7]:
            # apply eta projection filter to sensitivities
            dx = beta * (1 - np.tanh(beta * (xPrint - eta))**2) /\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
            dc[:] = dc * dx 
            dconstrs[:] = dconstrs * dx[:,None]
            # apply AM filter to sensitivities
            dc[:] = AMfilter(xTilde.reshape((nelx, nely)).T,
                            baseplate,
                            dc.reshape(nelx,nely).T[:,:,None]).T.flatten()
            dconstrs = AMfilter(xTilde.reshape((nelx, nely)).T,
                            baseplate,
                            np.transpose(dconstrs.reshape(nelx,nely,n_constr),
                                         (1,0,2)))
            dconstrs = np.transpose(dconstrs,(1,0,2)).reshape(nelx*nely,n_constr)
            if debug:
                print("Intermediate-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dconstrs: {2:.10f}".format(
                       loop, 
                       np.max(dc),
                       np.min(dconstrs)))
            # apply density filter to sensitivities
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dconstrs[:] = np.asarray(H*(dconstrs/Hs))
        elif ft in [-1]:
            pass
        if debug:
            print("Post-Sensitivity Filter: it.: {0}, dc: {1:.10f}, dconstrs: {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dconstrs)))
        #import sys 
        #sys.exit()
        # density update by solver
        xold[:] = x
        # optimality criteria
        if solver=="oc" and ft in filters:
             (x[:], g) = oc(nelx, nely, x, volfrac, 
                            dc, dconstrs[:,0], g, 
                            pass_el,
                            None,None,None,None,None,debug)
        elif solver=="oc" and ((ft in projections) or ft in [6]):
            (x[:],xTilde[:],xPhys[:],g) = oc(nelx, nely, x, volfrac, 
                                             dc, dconstrs[:,0], g, baseplate,
                                             pass_el,
                                             H,Hs,beta,eta,ft,
                                             debug=debug) 
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
        # globally convergent method of moving asymptotes, implementation by 
        # Arjen Deetman
        elif solver=="gcmma":
            mu0 = 1.0 # Scale factor for objective function
            mu1 = 1.0 # Scale factor for volume constraint function
            f0val = mu0*obj 
            df0dx = mu0*dc[np.newaxis].T
            fval = mu1*np.array([[xPhys.sum()/x.shape[0]-volfrac]])
            dfdx = mu1*(dconstrs/(x.shape[0]*volfrac))[np.newaxis]
            xval = x.copy()[np.newaxis].T
            #
            low,upp,raa0,raa= \
                  asymp(loop,x.shape[0],xval,xold1,xold2,xmin,xmax,low,upp,
                        raa0,raa,raa0eps,raaeps,df0dx,dfdx)
            xmma,ymma,zmma,lam,xsi,eta_gcmma,mu,zet,s,f0app,fapp= \
                  gcmmasub(n_constr,x.shape[0],loop,epsimin,xval,xmin,xmax,low,upp,
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
            # check if approximations conservative
            conserv = concheck(n_constr,epsimin,f0app,obj,fapp,constrs)
            # inner iterations
            if conserv == 0:
                for innerit in np.arange(ninneriter):
                    # update raa0 and raa
                    raa0,raa = raaupdate(xmma,xval,xmin,xmax,low,upp,
                                         obj,constrs,f0app,fapp,raa0, 
                                         raa,raa0eps,raaeps,epsimin)
                    # solve subproblem with new raa0 and raa:
                    xmma,ymma,zmma,lam,xsi,eta_gcmma,mu,zet,s,f0app,fapp = gcmmasub(n_constr,
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
                    raise ValueError("Here recalculate constraints again")
                    # It is checked if the approximations have become conservative:
                    conserv = concheck(n_constr,epsimin,f0app,obj,fapp,constrs)
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
            print("Post Density Update: it.: {0}, dc: {1:.10f}, dconstrs {2:.10f}".format(
                   loop, 
                   np.max(dc),
                   np.min(dconstrs)))
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
            # get volume preserving eta
            eta = find_eta(eta, xTilde, beta, volfrac)
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xTilde - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        elif ft in [7]:
            xTilde = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
            # filter needs the densities assembled to rectangular domain
            xPrint = AMfilter(xTilde.reshape((nelx, nely)).T, baseplate)
            xPrint = xPrint.T.flatten()
            # get volume preserving eta
            result = minimize(find_eta, x0=eta,
                              bounds=[[0., 1.]], 
                              method='Nelder-Mead',jac=True,tol=1e-10,
                              args=(xPrint,beta,volfrac))
            if result.success :
                eta = result.x
            else:
                raise ValueError("volume conserving eta could not be found: ",result)
            # eta projection
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xPrint - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        # Compute the change by the inf. norm
        change = (np.abs(x-xold)).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if debug == 1:
            if ft in projections or ft in [7]:
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
        elif (ft in projections+[7]) and (beta < 512) and \
            (loopbeta >= 50 or change < 0.01):
            beta = 2 * beta
            loopbeta = 0
            logging.info(f"Parameter beta increased to {beta}")
        elif (ft in projections+[7]) and (beta >= 512) and (change < 0.01):
            break
    # 
    plt.show()
    logging.shutdown()
    input("Press any key...")
    #
    if ft in [0,1]:
        xTilde=None
    export_vtk(filename="topopt_manufact", 
               nelx=nelx,nely=nely, 
               xPhys=xPhys,x=x, xTilde=xTilde,
               u=u,f=f,volfrac=volfrac)
    return x, obj
    
def oc(nelx, nely, x, volfrac, dc, dv, g, baseplate, pass_el,
       H,Hs,beta,eta,ft,
       debug=False):
    """
    Optimality criteria method (section 2.2 in paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier.
    Usually more sophisticated methods are used like Sequential Linear/Quadratic
    Programming or the Method of Moving Asymptotes.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    pass_el : None or np.array 
        array who contains indices used for un/masking passive elements. 0 
        means an active element that is part of the optimization, 1 and 2 
        indicate empty and full elements which are not part of the 
        optimization.

    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    l1 = 0
    l2 = 1e9
    if ft is None or ft in [0,1]:
        move = 0.2
        tol = 1e-3
    else:
        move = 0.05
        tol = 1e-5
    # reshape to perform vector operations
    xnew = np.zeros(nelx*nely)
    xTilde = np.zeros(nelx*nely)
    xPhys = np.zeros(nelx*nely)
    if debug:
        i = 0
    while (l2-l1)/(l1+l2) > tol and np.abs(l2-l1) > 1e-10:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, 
                             np.maximum(x-move, 
                                        np.minimum(1.0, 
                                                   np.minimum(x+move, 
                                                              x*np.sqrt(-dc/(dv)/lmid)))))
        #
        if ft in projections or ft in [6]:
            xTilde = np.asarray(H*xnew[np.newaxis].T/Hs)[:, 0]
        if ft in [2]:
            xPhys = 1 - np.exp(-beta*xTilde) + xTilde*np.exp(-beta)
        elif ft in [3]:
            xPhys = np.exp(-beta*(1-xTilde)) - (1-xTilde)*np.exp(-beta)
        elif ft in [4]:
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xTilde - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        elif ft in [6]:
            # filter needs the densities assembled to rectangular domain
            xPhys = AMfilter(xTilde.reshape((nelx, nely)).T, baseplate)
            xPhys = xPhys.T.flatten()
        else:
            xPhys = xnew
        # passive element update
        if pass_el is not None:
            xPhys[pass_el==1] = 0
            xPhys[pass_el==2] = 1
        #
        gt=g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        if debug == 2:
            i = i+1
            print("oc it.: {0} , l1: {1:.10f} l2: {2:.10f}, gt: {3:.10f}".format(
                   i, l1, l2, gt),
                  "x: {0:.10f} xTilde: {1:.10f} xPhys: {2:.10f}".format(
                    np.median(x),np.median(xTilde),np.median(xPhys)),
                  "dc: {0:.10f} dv: {1:.10f}".format(
                    np.max(dc),np.min(dv)))
            if np.isnan(gt):
                print()
                import sys 
                sys.exit()
    if ft in projections or ft in [6]:
        return (xnew, xTilde, xPhys, gt)
    else:
        return (xnew, gt)

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60  # 180
    nely = int(nelx/3)  # 60
    volfrac = 0.5  # 0.4
    rmin = 0.04*nelx  # 5.4
    penal = 3.0
    ft = 7 # ft==0 -> sens, ft==1 -> dens
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
             manufact = None,baseplate="S",
             passive=False,pde=False,solver="mma",
             nouteriter=2000,
             ninneriter=0,
             debug=False)
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.shutdown()
