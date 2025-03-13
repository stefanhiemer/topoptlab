import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,factorized
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely,
         nsteps,dt, 
         volfrac,penal,rmin,ft,
         solver="lu"):
    """
    Topology optimization with transient heat conduction to minimize 
    sum_{i=1}^{N_{t}} f_{i}^T u_{i}. At the moment this function is purely for
    demonstration purposes and the objective is not intended to be sensible. 
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nsteps : int
        number of timesteps.
    dt : float
        timestep. With default arguments, values above 0.25 should be fine. If 
        one goes much larger than that, one should check for stability/sensible
        solutions.
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
    solver : str
        scipy solver for the FEM and adjoint problem. Either "direct" or "lu".

    Returns
    -------
    None.

    """
    print("Minimum compliance problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # Max and min heat conduction
    kmin = 1e-5
    kmax = 1.0
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
    iE = np.tile(edofMat,KE.shape[0]).flatten()
    jE = np.repeat(edofMat,KE.shape[0]).flatten()  
    # assemble filter
    H,Hs = assemble_filter(rmin=rmin,el=el,nelx=nelx,nely=nely)
    # BC's
    dofs = np.arange(ndof)
    # heat sink
    fixed = []
    fixed = np.arange(int(nely / 2 + 1 - nely / 20), 
                      int(nely / 2 + 1 + nely / 20) + 1)
    # general
    free = np.setdiff1d(dofs, fixed)
    # Solution and RHS vectors
    f = np.zeros((ndof, nsteps))
    u = np.zeros((ndof, nsteps+1))
    # load/source
    #u[0,0] = 100 # peak
    f[:, :] = -1 # constant source
    # Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
                   interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    fig.show()
    # Set loop counter and gradient vectors 
    loop=0
    change=1
    dv = np.ones(nely*nelx)
    dc = np.ones(nely*nelx)
    while change>0.01 and loop<2000:
        loop=loop+1
        # Setup stiffness and mass matrix
        sK=((KE.flatten()[:,None])*(kmin+(xPhys)\
                **penal*(kmax-kmin))).flatten(order='F')
        K = coo_matrix((sK,(iE,jE)),shape=(ndof,ndof)).tocsc()
        sM = ((ME.flatten()[np.newaxis]).T*(xPhys**penal)).flatten(order='F')
        M = coo_matrix((sM,(iE,jE)),shape=(ndof,ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
        M = M[free,:][:,free]
        # Solve system
        if solver == "lu":
            lu = factorized(M/dt + K) # LU decomposition. returns a function
        for nt in np.arange(1,nsteps+1):
            if solver == "lu":
                u[free,nt]=lu(f[free,nt-1] + (M/dt).dot(u[free,nt-1]))
            elif solver == "direct":
                u[free,nt]=spsolve(M/dt + K,
                                   f[free,nt-1] + (M/dt).dot(u[free,nt-1]))
            #im.set_array(u[:,nt].reshape((nelx+1,nely+1)).T+1e-6)
            #fig.canvas.draw()
            #plt.pause(0.01)
        # Objective
        obj = (f * u[:,1:]).sum()
        # sensitivity of objective/constraints
        # here the derivative of the cost function with regards to explicit 
        # occurences of design variables/element densities in the obj. function 
        # would occur. In this special case, design variables do not explicitly 
        # appear in the obj. and are implicitly "hidden" in the state variables.
        dc[:] = 0 
        h = np.zeros((ndof,nsteps)) # adjoint vectors/Lagrangian multipliers
        if solver == "lu":
            h[free,-1] = lu(-f[free,-1])
        elif solver == "direct":
            h[free,-1] = spsolve(M/dt + K, -f[free,-1])
        dc[:] += ((kmax-kmin)*np.dot(h[edofMat,-1], KE)*\
                   u[edofMat,-1]).sum(1)+\
                 (np.dot(h[edofMat,-1], ME)*\
                  (u[edofMat,-1]-u[edofMat,-2])/dt).sum(1)
        for nt in np.arange(nsteps-1)[-1::-1]: 
            # adjoint problem
            if solver == "lu":
                h[free,nt] = lu(-f[free,nt] + (M/dt).dot(h[free,nt+1]))
            elif solver == "direct":
                h[free,nt] = spsolve(M/dt + K,
                                     -f[free,nt] + (M/dt).dot(h[free,nt+1]))
            # update gradient
            dc[:] += ((kmax-kmin)*np.dot(h[edofMat,nt], KE)*\
                       u[edofMat,nt+1]).sum(1)+\
                     (np.dot(h[edofMat,nt], ME)*\
                      (u[edofMat,nt+1]-u[edofMat,nt])/dt).sum(1)
        dc[:] *= penal*xPhys**(penal-1)
        # finite differences test
        """
        dp = 1e-9
        _dc = np.zeros(xPhys.shape)
        _u = u = np.zeros((ndof, nsteps+1))
        for i in np.arange(xPhys.shape[0]):
            _xPhys = xPhys.copy()
            _xPhys[i] += dp 
            # Setup stiffness and mass matrix
            _sK=((KE.flatten()[:,None])*(kmin+(_xPhys)\
                    **penal*(kmax-kmin))).flatten(order='F')
            _K = coo_matrix((_sK,(iE,jE)),shape=(ndof,ndof)).tocsc()
            _sM = ((ME.flatten()[np.newaxis]).T*(_xPhys**penal)).flatten(order='F')
            _M = coo_matrix((_sM,(iE,jE)),shape=(ndof,ndof)).tocsc()
            # Remove constrained dofs from matrix
            _K = _K[free,:][:,free]
            _M = _M[free,:][:,free]
            for nt in np.arange(1,nsteps+1):
                _u[free,nt]=spsolve(_M/dt + _K,
                                    f[free,nt-1] + (_M/dt).dot(u[free,nt-1]))
                #im.set_array(u[:,nt].reshape((nelx+1,nely+1)).T+1e-6)
                #fig.canvas.draw()
                #plt.pause(0.01)
            # Objective
            _obj= (f*_u[:,1:]).sum()
            _dc[i] = (_obj-obj)/dp
        print(dc)
        print(_dc)
        import sys 
        sys.exit()
        """
        # sensitivity analysis of constraints that are unaffacted by time
        # this means in 99.9 % of the cases constraints purely on the densities
        dv[:] = np.ones(nely*nelx)
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        # Optimality criteria
        xold[:]=x
        x[:],g=oc(x,volfrac,dc,dv,g)
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
        print("it.: {0} , obj.: {1:.8f} Vol.: {2:.8f}, ch.: {3:.8f}"\
              .format(loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
    # Make sure the plot stays and that the shell remains    
    plt.show()
    input("Press any key...")
    return
# matrix filter
def assemble_filter(rmin,el,nelx,nely):
    """
    Assemble matrix filter.
    
    Parameters
    ----------
    rmin : float
        filter radius measured by number of elements.
    el : np.array, shape (nel)
        element indices.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
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
    fac = rmin-np.sqrt((i-k)**2+(j-l)**2)
    iH[cc] = el # row
    jH[cc] = k*nely+l #column
    sH[cc] = np.maximum(0.0, fac)
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)
    return H,Hs
#element stiffness matrix
def lk():
    """
    Create element stiffness matrix for 2D transient heat equation with 
    bilinear quadrilateral elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[2/3, -1/6, -1/3, -1/6,],
                     [-1/6, 2/3, -1/6, -1/3],
                     [-1/3, -1/6, 2/3, -1/6],
                     [-1/6, -1/3, -1/6, 2/3]])
#element mass matrix
def lm():
    """
    Create element mass matrix for 2D transient heat equation with bilinear 
    quadrilateral elements.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[1/9, 1/18, 1/36, 1/18],
                     [1/18, 1/9, 1/18, 1/36],
                     [1/36, 1/18, 1/9, 1/18],
                     [1/18, 1/36, 1/18, 1/9]])
# Optimality criterion
def oc(x,volfrac,dc,dv,g):
    """
    Optimality criteria method (section 2.2 in top88 paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Overtaken and adapted from the 
    
    165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN
    
    Parameters
    ----------
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
    l1=0
    l2=1e9
    move = 0.1#0.1
    #damp = 0.3
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew[:]= np.maximum(0.0,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/dv/lmid)))))
        #xnew[:] = np.maximum(0.0, np.maximum(
        #    x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10,
        #                                                           -dc/dv/lmid)**damp))))
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
    volfrac = 0.4
    rmin = 1.2
    penal = 3.0
    ft=1 # ft==0 -> sens, ft==1 -> dens
    dt = 4e0 # 1/(4 * kmax) is a lower bound
    nsteps = 250
    solver="lu"
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    if len(sys.argv)>7: dt     =int(sys.argv[7])
    if len(sys.argv)>8: nsteps =int(sys.argv[8])
    if len(sys.argv)>9: solver =int(sys.argv[9])
    main(nelx,nely,nsteps,dt,solver,volfrac,penal,rmin,ft)
