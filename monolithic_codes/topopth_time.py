# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# minor modifications by Stefan Hiemer (January 2025)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft):
    """
    Topology optimization for maximum stiffness with the SIMP method based on 
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

    Returns
    -------
    None.

    """
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # Max and min heat conduction
    kmin = 1e-2
    kmax = 1.0
    dt = 1 # 1/(4 * kmax) is like a lower bound
    nsteps = 100
    # dofs:
    ndof = (nelx+1)*(nely+1)
    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float)
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    dc=np.zeros((nely,nelx), dtype=float)
    # build stiffness matrix
    # FE: Build the index vectors for the for coo matrix format.
    KE = lk()
    ME = lm()
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((n1+1, n2+1, n2, n1))
    # Construct the index pointers for the coo format
    iE = np.kron(edofMat, np.ones((4, 1))).flatten()
    jE = np.kron(edofMat, np.ones((1, 4))).flatten()
    # Filter: Build (and assemble) the index+data vectors for the coo matrix format
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
    # BC's
    dofs = np.arange(ndof)
    # heat sink
    fixed = []
    #fixed = np.arange(int(nely / 2 + 1 - nely / 20), 
    #                  int(nely / 2 + 1 + nely / 20) + 1)
    # general
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, nsteps))
    u = np.zeros((ndof, nsteps+1))
    # load/source
    u[0,0] = 100
    #f[:, 0] = -1 # constant source
    # Initialize plot and plot the initial design
    #plt.ion() # Ensure that redrawing is possible
    #fig,ax = plt.subplots()
    #im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
    #               interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    #ax.tick_params(axis='both',
    #               which='both',
    #               bottom=False,
    #               left=False,
    #               labelbottom=False,
    #               labelleft=False)
    #fig.show()
    # Set loop counter and gradient vectors 
    loop=0
    change=1
    dv = np.ones(nely*nelx)
    dc = np.ones(nely*nelx)
    ce = np.ones(nely*nelx)
    while change>0.01 and loop<2000:
        loop=loop+1
        # Setup stiffness and mass matrix
        sK=((KE.flatten()[:,None])*(kmin+(xPhys)\
                **penal*(kmax-kmin))).flatten(order='F')
        K = coo_matrix((sK,(iE,jE)),shape=(ndof,ndof)).tocsc()
        sM = sK=((ME.flatten()[np.newaxis]).T*(xPhys**penal)).flatten(order='F')
        M = coo_matrix((sM,(iE,jE)),shape=(ndof,ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
        M = M[free,:][:,free]
        # Solve system
        """
        plt.ion()
        fig,ax = plt.subplots()
        im = ax.imshow(u[:,0].reshape((nelx+1,nely+1)).T, cmap='gray',\
                       interpolation='none',
                       norm=colors.LogNorm(vmin=u[:,0].min(),vmax=u[:,0].max()))
        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        #fig.show()
        """
        for nt in np.arange(1,nsteps+1):
            u[free,nt]=spsolve(M/dt + K,
                               f[free,nt-1] + (M/dt).dot(u[free,nt-1]))
            #im.set_array(u[:,nt].reshape((nelx+1,nely+1)).T+1e-6)
            #fig.canvas.draw()
            #plt.pause(0.01)
        # Objective and sensitivity
        ce[:] = 0
        for nt in np.arange(1,nsteps+1):
            ce[:] = ce[:] + (np.dot(u[edofMat,nt].reshape(nelx*nely,4),KE) * \
                                    u[edofMat,nt].reshape(nelx*nely,4) ).sum(1)
        obj=( (kmin+xPhys**penal*(kmax-kmin))*ce ).sum()
        break
        dc[:]=(-penal*xPhys**(penal-1)*(kmax-kmin))*ce
        dv[:] = np.ones(nely*nelx)
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        # Optimality criteria
        xold[:]=x
        x[:],g=oc(nelx,nely,x,volfrac,dc,dv,g)
        # Filter design variables
        if ft==0:   
            xPhys[:]=x
        elif ft==1:    
            xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        # Compute the change by the inf. norm
        change=(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1)).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nelx,nely)).T)
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}"\
              .format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    # Make sure the plot stays and that the shell remains    
    plt.show()
    input("Press any key...")
    return
#element stiffness matrix
def lk():
    return np.array([[2/3, -1/6, -1/3, -1/6,],
                     [-1/6, 2/3, -1/6, -1/3],
                     [-1/3, -1/6, 2/3, -1/6],
                     [-1/6, -1/3, -1/6, 2/3]])
#element mass matrix
def lm():
    return np.array([[1/9, 1/18, 1/36, 1/18],
                     [1/18, 1/9, 1/18, 1/36],
                     [1/36, 1/18, 1/9, 1/18],
                     [1/18, 1/36, 1/18, 1/9]])
# Optimality criterion
def oc(nelx,nely,x,volfrac,dc,dv,g):
    l1=0
    l2=1e9
    move=0.2
    # reshape to perform vector operations
    xnew=np.zeros(nelx*nely)
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return xnew,gt
# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx = 40
    nely = 40
    volfrac = 1.0
    rmin = 1.2
    penal = 3.0
    ft=1 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    main(nelx,nely,volfrac,penal,rmin,ft)
