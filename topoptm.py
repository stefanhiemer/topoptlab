# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
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
    print("Compliant mechanism problem with OC")
    print("nodes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " +
          str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", "Density based",
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
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
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
    fixed = np.union1d(np.arange(1,(nelx+1)*(nely+1)*2,(nely+1)*2), # symmetry
                       np.arange(2*(nely+1)-4,2*(nely+1))) # bottom left bit
    din = 0
    dout = 2*nelx*(nely+1)
    # general
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 2))
    u = np.zeros((ndof, 2))
    # Set load
    f[din,0] = 1
    f[dout,1] = -1
    # get rid of fixed degrees of freedom from stiffness matrix 
    mask = ~(np.isin(iK,fixed) | np.isin(jK,fixed))
    #
    _din, _dout = update_spring(np.array([din,dout]), fixed, 
                                np.isin(np.array([din,dout]),fixed))
    iK,jK = update_stiff(iK, fixed, mask),update_stiff(jK, fixed, mask)
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
        sK = (KE.flatten()[:,None]*(Emin+(xPhys)
              ** penal*(Emax-Emin))).flatten(order='F')[mask]
        K = coo_matrix((sK, (iK, jK)), shape=(ndof_free, ndof_free)).tocsc()
        # add springs to stiffness matrix
        K[_din,_din] += kin # 
        K[_dout,_dout] += kout # 
        # Solve systems
        u[free, :] = spsolve(K, f[free, :])
        # Objective and sensitivity
        obj = u[dout,0].copy()
        dc = penal*xPhys**(penal-1)*(Emax-Emin)*(np.dot(u[edofMat,1].reshape(nelx*nely, 8), KE)*\
              u[edofMat,0].reshape(nelx*nely, 8)).sum(1)
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
            print("it.: {0} , obj.: {1:.8f} Vol.: {2:.8f}, ch.: {3:.8f}".format(
            loop, obj, xPhys.mean(), change))
        # convergence check
        if change < 0.01:
            break 
    #
    plt.show()
    input("Press any key...")
    #
    export_vtk(filename="topoptm.vtk", 
               nelx=nelx,nely=nely, 
               xPhys=xPhys,x=x, 
               u=u,f=f,volfrac=volfrac)
    return x, obj

def update_stiff(indices,fixed,mask):
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
    move = 0.1
    damp = 0.3
    # reshape to perform vector operations
    xnew = np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10,
                                                                    -dc/dv/lmid)**damp))))
        
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

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 20
    volfrac = 0.3
    rmin = 1.2#0.04*nelx  # 5.4
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
