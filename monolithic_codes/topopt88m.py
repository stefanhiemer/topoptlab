# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# minor modifications by Stefan Hiemer (January 2025)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,factorized
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft,solver="lu"):
    """
    Topology optimization for force inverter with the SIMP method based on 
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
    solver : str
        scipy solver for the FEM and adjoint problem. Either "direct" or "lu".

    Returns
    -------
    None.

    """
    print("Force Inverter problem with OC")
    print("ndes: " + str(nelx) + " x " + str(nely))
    print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
    print("Filter method: " + ["Sensitivity based","Density based"][ft])
    # Max and min stiffness
    Emin=1e-9
    Emax=1.0
    # stiffness constants springs 
    kin = 0.1
    kout = 0.1
    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float,order="F")
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    # fetch element stiffness matrix
    KE = lk()
    # # dofs:
    ndof = int(KE.shape[-1]/4) *(nelx+1)*(nely+1)
    # FE: Build the index vectors for the for coo matrix format.
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMat = np.column_stack((2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 
                               2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # Construct the index pointers for the coo format
    iK = np.tile(edofMat,KE.shape[-1]).flatten()
    jK = np.repeat(edofMat,KE.shape[-1]).flatten()   
    # assemble filter
    H,Hs = assemble_filter(rmin=rmin,el=el,nelx=nelx,nely=nely)
    # BC's and support
    dofs = np.arange(ndof)
    fixed = np.union1d(np.arange(1,(nelx+1)*(nely+1)*2,(nely+1)*2), # symmetry
                       np.arange(2*(nely+1)-4,2*(nely+1))) # bottom left bit
    din = 0
    dout = 2*nelx*(nely+1)
    # Solution, RHS and adjoint vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    h = np.zeros((ndof,1))
    # Set load
    f[din,0] = 1
    # general
    free = np.setdiff1d(dofs, fixed)
    # indicator array for the output node and later for the adjoint problem
    l = np.zeros((ndof, 1))
    l[dout,0] = 1
    # Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(-xPhys.reshape((nely,nelx),order="F"), cmap='gray',
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
        # Setup and solve FE problem
        sK=((KE.flatten()[np.newaxis]).T*(Emin+(xPhys)**penal*(Emax-Emin))).flatten(order='F')
        K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
        # add springs to stiffness matrix
        K[din,din] += kin # 
        K[dout,dout] += kout # 
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
        # Solve system 
        if solver == "lu":
            lu = factorized(K)
            u[free,0]=lu(f[free,0])
        elif solver == "direct":
            u[free,0]=spsolve(K,f[free,0])    
        # Objective and sensitivity
        obj = u[l[:,0]!=0].sum()
        if solver == "lu":
            h[free,0] = lu(-l[free,0])
        elif solver == "direct":
            h[free,0] = spsolve(K,-l[free,0])
        dc[:]= penal*xPhys**(penal-1)*(Emax-Emin)*(np.dot(h[edofMat,0], KE)*\
               u[edofMat,0]).sum(1)
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
        change=np.abs(x-xold).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nely,nelx),order="F"))
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(\
                    loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
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
    Create element stiffness matrix for 2D linear elasticity equation with 
    bilinear quadrilateral elements in plane stress. Taken from the 88 line code.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
    E=1
    nu=0.3
    k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,
                -1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE
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
    move=0.1
    damp = 0.3
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid=0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10,
                                                                    -dc/dv/lmid)**damp))))
        gt=g+np.sum((dv*(xnew-x)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return xnew,gt
# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=40
    nely=20
    volfrac=0.3
    rmin=1.2
    penal=3.0
    ft=0 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    main(nelx,nely,volfrac,penal,rmin,ft)
