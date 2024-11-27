# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
from os.path import isfile
from os import remove
import logging 

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from topoptlab.optimality_criterion import oc_top88
from topoptlab.output_designs import export_vtk

message = "MMA module not found. Get it from https://github.com/arjendeetman/GCMMA-MMA-Python/tree/master .Copy mma.py in the same directory as this python file."
try:
    from mmapy import mmasub,gcmmasub,asymp,concheck,raaupdate
except ModuleNotFoundError:
    raise ModuleNotFoundError(message)
    

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         pde=False, passive=False, solver="oc", 
         nouteriter=2000, ninneriter=15,
         display=True,export=True,write_log=True):
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
        if isfile("topopt.log"):
            remove("topopt.log")
        logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler("topopt.log"),
                            logging.StreamHandler()])
        #
        logging.info(f"Minimum compliance problem with {solver}")
        logging.info(f"nodes: {nelx} x {nely}")
        logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
        logging.info("Filter method: " + ["Sensitivity based", "Density based",
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
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
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
    dv = np.ones(nelx*nely)
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
                ce = (np.dot(ui[edofMat].reshape(nelx*nely, KE.shape[0]), KE)
                         * ui[edofMat].reshape(nelx*nely, KE.shape[0])).sum(1)
                obj += ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
                dc[:] -= penal*xPhys**(penal-1)*(Emax-Emin)*ce
        dv[:] = np.ones(nely*nelx)
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
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0]
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
            dv[:] = TF.T @ spsolve(KF,TF@dv)
        elif ft == -1:
            pass
        # density update by solver
        xold[:] = x
        # optimality criteria
        if solver=="oc":
            (x[:], g) = oc_top88(nelx, nely, x, volfrac, dc, dv, g, pass_el)
        # method of moving asymptotes, implementation by Arjen Deetman
        elif solver=="mma":
            xval = x.copy()[np.newaxis].T
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = update_mma(x,xold1,xold2,xPhys,obj,dc,dv,loop,
                                                                     m,xmin,xmax,low,upp,a0,a,c,d,move)
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
            xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp= \
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
                #xPhys[:] = TF.T @ spsolve_triangular(LF.T, 
                #                                     (spsolve_triangular(LF, TF@x)),
                #                                     lower=False)
                #xPhys[:] = TF.T @ LU.solve(TF@x)
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
                    xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,f0app,fapp = gcmmasub(m,
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
                        #xPhys[:] = TF.T @ spsolve_triangular(LF.T, 
                        #                                     (spsolve_triangular(LF, TF@x)),
                        #                                     lower=False)
                        #xPhys[:] = TF.T @ LU.solve(TF@x)
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
            xPhys[:]  = x
        # Compute the change by the inf. norm
        change = (np.abs(x-xold)).max()
        # Plot to screen
        if display:
            im.set_array(-xPhys.reshape((nelx, nely)).T)
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
                   nelx=nelx,nely=nely, 
                   xPhys=xPhys,x=x, 
                   u=u,f=f,volfrac=volfrac)
    return x, obj


def update_indices(indices,fixed,mask):
    """
    Update the indices for the stiffness matrix construction by kicking out
    the fixed degrees of freedom and renumbering the indices.

    Parameters
    ----------
    indices : np.array
        indices of degrees of freedom used to construct the stiffness matrix.
    fixed : np.array
        indices of fixed degrees of freedom.
    mask : np.array
        mask to kick out fixed degrees of freedom.

    Returns
    -------
    indices : np.arrays
        updated indices.

    """
    val, ind = np.unique(indices,return_inverse=True)
    
    _mask = ~np.isin(val, fixed)
    val[_mask] = np.arange(_mask.sum())
    
    return val[ind][mask]

def lk():
    """
    Create element stiffness matrix.
    
    Returns
    -------
    Ke : np.array, shape (8,8)
        element stiffness matrix.
        
    """
    E = 1
    nu = 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return (KE)

def update_mma(x,xold1,xold2,xPhys,obj,dc,dv,iteration,
               m,xmin,xmax,low,upp,a0,a,c,d,move):
    mu0 = 1.0 # Scale factor for objective function
    mu1 = 1.0 # Scale factor for volume constraint function
    f0val = mu0*obj 
    df0dx = mu0*dc[np.newaxis].T
    fval = mu1*np.array([[xPhys.mean()-volfrac]])
    dfdx = mu1*(dv/(x.shape[0]*volfrac))[np.newaxis]
    xval = x.copy()[np.newaxis].T 
    #print(f0val.shape)
    #print(df0dx.shape)
    #print(fval.shape)
    #print(dfdx.shape)
    #print(xval.shape)
    #raise ValueError
    return mmasub(m,x.shape[0],iteration,xval,xmin,xmax,xold1,xold2,f0val,df0dx,
                  fval,dfdx,low,upp,a0,a,c,d,move)