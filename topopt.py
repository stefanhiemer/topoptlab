# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
from __future__ import division
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx, nely, volfrac, penal, rmin, ft):
    """
    Topology optimization workflow with the SIMP method.
    
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
        1 density filtering, 2 no filtering

    Returns
    -------
    None.

    """
    print("Minimum compliance problem with OC")
    print("nodes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " +
          str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based", 
                               "Density based",
                               "No filter"][ft])
    # Max and min stiffness
    Emin = 1e-9
    Emax = 1.0
    # dofs:
    ndof = 2*(nelx+1)*(nely+1)
    # Allocate design variables (as array), initialize and allocate sens.
    x = volfrac * np.ones(nely*nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()
    g = 0  # must be initialized to use the NGuyen/Paulino OC approach
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat[el.flatten(), :] = np.column_stack((2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 
                                      2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # Construct the index pointers for the coo format
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    
    kk1 = np.maximum(np.floor(el/nely)-(np.ceil(rmin)-1), 0).astype(int)
    kk2 = np.minimum(np.floor(el/nely)+np.ceil(rmin), nelx).astype(int)
    ll1 = np.maximum(el%nely-(np.ceil(rmin)-1), 0).astype(int)
    ll2 = np.minimum(el%nely+np.ceil(rmin), nely).astype(int)
    
    for i in el:
        j=i%nely
        i=np.floor(i/nely).astype(int)
        for k in range(kk1[i], kk2[i]):
            for l in range(ll1[i], ll2[i]):
                col = k*nely+l
                fac = rmin-np.sqrt(((i-k)**2+(j-l)**2))
                iH[cc] = el[i]
                jH[cc] = col
                sH[cc] = np.maximum(0.0, fac)
                cc = cc+1
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)
    print(Hs)
    import sys 
    sys.exit()
    # BC's and support
    dofs = np.arange(2*(nelx+1)*(nely+1))
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # Set load
    f[1, 0] = -1
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx, nely)).T, cmap='gray',
                   interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
    fig.show()
    # Set loop counter and gradient vectors
    for loop in np.arange(2000):
        # Setup and solve FE problem
        sK = ((KE.flatten()[None]).T*(Emin+(xPhys)
              ** penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free, :][:, free]
        # Solve system
        u[free, 0] = spsolve(K, f[free, 0])
        #u[free, 0] = cg(K, f[free, 0],x0=)
        # Objective and sensitivity
        ce = (np.dot(u[edofMat].reshape(nelx*nely, 8), KE)
                 * u[edofMat].reshape(nelx*nely, 8)).sum(1)
        obj = ((Emin+xPhys**penal*(Emax-Emin))*ce).sum()
        dc = (-penal*xPhys**(penal-1)*(Emax-Emin))*ce
        dv = np.ones(nely*nelx)
        # Sensitivity filtering:
        if ft == 0:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T /
                               Hs)[:, 0] / np.maximum(0.001, x)
        elif ft == 1:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:, 0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:, 0]
        elif ft == 2:
            pass
        # Optimality criteria
        xold[:] = x
        (x[:], g) = oc(nelx, nely, x, volfrac, dc, dv, g)
        # Filter design variables
        if ft == 0:
            xPhys[:] = x
        elif ft == 1:
            xPhys[:] = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
        elif ft == 2:
            xPhys[:] = x
        # Compute the change by the inf. norm
        change = np.linalg.norm(
            x.reshape(nelx*nely, 1)-xold.reshape(nelx*nely, 1), np.inf)
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx, nely)).T)
        fig.canvas.draw()
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(
            loop, obj, (g+volfrac*nelx*nely)/(nelx*nely), change))
        # convergence check
        if change < 0.01:
            break
    # Make sure the plot stays and that the shell remains
    #plt.show()
    #input("Press any key...")
# element stiffness matrix


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
# Optimality criterion


def oc(nelx, nely, x, volfrac, dc, dv, g):
    """
    Optimality criteria method (section 2.2 in paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Usually more sophisticated methods are 
    used like Sequential Linear/Quadratic Programming or the Method of Moving 
    Asymptotes.
    
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
        gt = g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return (xnew, gt)


# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60  # 180
    nely = 20  # 60
    volfrac = 0.5  # 0.4
    rmin = 2.4  # 5.4
    penal = 3.0
    ft = 0  # ft==0 -> sens, ft==1 -> dens
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
    main(nelx, nely, volfrac, penal, rmin, ft)
