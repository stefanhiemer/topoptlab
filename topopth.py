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

from output_designs import export_vtk

# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft, 
         pde=False, passive=False, verbose=True):
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

    Returns
    -------
    None.

    """
    # check if log file exists and if true delete
    if isfile("topopth.log"):
        remove("topopth.log")
    logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler("topopth.log"),
                        logging.StreamHandler()])
    logging.info("Minimum compliance problem with oc")
    logging.info(f"nodes: {nelx} x {nely}")
    logging.info(f"volfrac: {volfrac}, rmin: {rmin},  penal: {penal}")
    logging.info("Filter method: " + ["Sensitivity based", "Density based",
                               "Haeviside","No filter"][ft])
    # Max and min stiffness
    kmin = 1e-3
    kmax = 1.0
    # dofs:
    ndof = (nelx+1)*(nely+1)
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely*nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((n1+1, n2+1, n2, n1))
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((4, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 4))).flatten()
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
    # BC's
    dofs = np.arange(ndof)
    # heat sink
    start = int(nely / 2 + 1 - nely / 20)
    end = int(nely / 2 + 1 + nely / 20)
    fixed = np.arange(start, end + 1)
    # general
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # load/source
    f[:, 0] = -1 # constant source
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
    # gradient for the volume constraint is constant regardless of iteration
    dv = np.ones(nely*nelx)
    # optimization loop
    for loop in np.arange(2000):
        # Setup and solve FE problem
        sK = (KE.flatten()[:,None]*(kmin+(xPhys)
              ** penal*(kmax-kmin))).flatten(order='F')[mask]
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
        dc = 0
        for i in np.arange(f.shape[1]):
            #ce = (np.dot(u[edofMat,i].reshape(nelx*nely, 8), KE)
            #         * u[edofMat,i].reshape(nelx*nely, 8)).sum(1)
            ui = u[:,i]
            ce = (np.dot(ui[edofMat], KE) * ui[edofMat]).sum(1)
            obj += ((kmin+xPhys**penal*(kmax-kmin))*ce).sum()
            dc -= penal*xPhys**(penal-1)*(kmax-kmin)*ce
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
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g, pass_el)
        
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
            pass
        # Compute the change by the inf. norm
        change = (np.abs(x-xold)).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        if verbose: 
            logging.info("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break
    #
    plt.show()
    input("Press any key...")
    #
    export_vtk(filename="topopth", 
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
    Ke : np.array, shape (4,4)
        element stiffness matrix.
        
    """
    KE = np.array([[2/3, -1/6, -1/3, -1/6,],
                   [-1/6, 2/3, -1/6, -1/3],
                   [-1/3, -1/6, 2/3, -1/6],
                   [-1/6, -1/3, -1/6, 2/3]])
    return (KE)

def oc(nelx, nely, x, volfrac, dc, dv, g, pass_el):
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
    move = 0.2
    # reshape to perform vector operations
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
        
        # passive element update
        if pass_el is not None:
            xnew[pass_el==1] = 0
            xnew[pass_el==2] = 1
        gt = xnew.sum() - volfrac * x.shape[0] #g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        
    return (xnew, gt)

def threshold(xPhys, volfrac):
    """
    Threshold grey scale design to black and white design.

    Parameters
    ----------
    xPhys : np.array, shape (nel)
        element densities for topology optimization used for scaling the 
        material properties. 
    volfrac : float
        volume fraction.

    Returns
    -------
    xPhys : np.array, shape (nel)
        thresholded element densities for topology optimization used for scaling the 
        material properties. 

    """
    indices = np.flip(np.argsort(xPhys))
    vt = np.floor(volfrac*xPhys.shape[0]).astype(int)
    xPhys[indices[:vt]] = 1.
    xPhys[indices[vt:]] = 0.
    print("Thresholded Vol.: {0:.3f}".format(vt/xPhys.shape[0]))
    return xPhys

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40
    volfrac = 0.4
    rmin = 1.2
    penal = 3.0
    ft = 0 # ft==0 -> sens, ft==1 -> dens
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
    main(nelx, nely, volfrac, penal, rmin, ft, passive=False,pde=False)
