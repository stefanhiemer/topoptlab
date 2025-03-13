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
    # Max and min stiffness
    Emin=1e-9
    Emax=1.0
    # Max and min heat conductivity
    kmin=1e-9
    kmax=1.0
    # 
    a = 0.05
    # stiffness constants springs 
    kin = 0.1
    kout = 0.1
    # Allocate design variables (as array), initialize and allocate sens.
    x=volfrac * np.ones(nely*nelx,dtype=float,order="F")
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    dc=np.zeros((nely,nelx), dtype=float)
    # fetch element stiffness matrix
    KeT = lkT()
    KeE = lkE(E=1.0,nu=0.3)
    KeET = lkET(E=1.0,nu=0.3,a=a)
    # # dofs:
    ndofT = int(KeT.shape[-1]/4) *(nelx+1)*(nely+1)
    ndofE = int(KeE.shape[-1]/4) *(nelx+1)*(nely+1)
    # FE: Build the index vectors for the for coo matrix format.
    elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
    el = np.arange(nelx*nely)
    n1 = ((nely+1)*elx+ely).flatten()
    n2 = ((nely+1)*(elx+1)+ely).flatten()
    edofMatT = np.column_stack((n1+1, n2+1, n2, n1))
    edofMatE = np.column_stack((2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 
                                2*n2, 2*n2+1, 2*n1, 2*n1+1))
    # Construct the index pointers for the coo format
    iKE = np.tile(edofMatE,KeE.shape[-1]).flatten()
    jKE = np.repeat(edofMatE,KeE.shape[-1]).flatten() 
    iKT = np.tile(edofMatT,KeT.shape[-1]).flatten()
    jKT = np.repeat(edofMatT,KeT.shape[-1]).flatten()
    iKET = np.tile(edofMatT,KeE.shape[-1]).flatten() 
    jKET = np.repeat(edofMatE,KeT.shape[-1]).flatten()
    # assemble filter
    H,Hs = assemble_filter(rmin=rmin,el=el,nelx=nelx,nely=nely)
    # Solution, RHS 
    q = np.zeros((ndofT, 1))
    T = np.zeros((ndofT, 1))
    # BC's for heat conduction
    dofsT = np.arange(ndofT)
    # heat sink
    fixedT = dofsT[-(nely+1)]#np.arange(nely+1)
    freeT = np.setdiff1d(dofsT, fixedT)
    hT = np.zeros((ndofT,1))
    # heat source
    q[:nely+1] = 1e-2
    #q[-(nely+1):, 0] = 1
    # BC's and support of linear elasticity
    dofsE = np.arange(ndofE)
    fixedE = np.union1d(np.arange(1,(nelx+1)*(nely+1)*2,(nely+1)*2), # symmetry
                        np.arange(2*(nely+1)-4,2*(nely+1))) # bottom left bit
    din = 0
    dout = 2*nelx*(nely+1)
    # Solution, RHS and adjoint vectors
    f = np.zeros((ndofE, 1))
    u = np.zeros((ndofE, 1))
    hE = np.zeros((ndofE,1))
    # Set load
    f[din,0] = 1
    # general
    freeE = np.setdiff1d(dofsE, fixedE)
    # indicator array for the output node and later for the adjoint problem
    l = np.zeros((ndofE, 1))
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
        # Setup and solve heat FE problem
        k = (kmin+(xPhys)**penal*(kmax-kmin))
        sKT=((KeT.flatten()[None]).T*k).flatten(order='F')
        K_T = coo_matrix((sKT,(iKT,jKT)),shape=(ndofT,ndofT)).tocsc()
        # Remove constrained dofs from matrix
        K_T = K_T[freeT,:][:,freeT]
        # Solve system for temperature 
        if solver == "lu":
            lu_T = factorized(K_T)
            T[freeT,0]=lu_T(q[freeT,0])
        elif solver == "direct":
            T[freeT,0]=spsolve(K_T,q[freeT,0])  
        # Setup and solve elastic FE problem
        E = (Emin+(xPhys)**penal*(Emax-Emin))
        sKE=((KeE.flatten()[np.newaxis]).T*E).flatten(order='F')
        K_E = coo_matrix((sKE,(iKE,jKE)),shape=(ndofE,ndofE)).tocsc()
        # add springs to stiffness matrix
        K_E[din,din] += kin
        K_E[dout,dout] += kout
        # Remove constrained dofs from matrix
        K_E = K_E[freeE,:][:,freeE]
        # create right hand side
        fTe = KeET@T[edofMatT]
        # assemble
        fT = np.zeros(f.shape)
        np.add.at(fT[:,0],
                  edofMatE.flatten(),
                  (E[:,None,None] * fTe).flatten())
        # Solve system 
        if solver == "lu":
            lu_E = factorized(K_E)
            u[freeE,0]=lu_E(f[freeE,0] + fT[freeE,0])
        elif solver == "direct":
            u[freeE,0]=spsolve(K_E,f[freeE,0] + fT[freeE,0])
        # Objective
        obj = u[l[:,0]!=0].sum()
        # first adjoint problem
        if solver == "lu":
            hE[freeE,0] = lu_E(-l[freeE,0])
        elif solver == "direct":
            hE[freeE,0] = spsolve(K_E,-l[freeE,0])
        # set up second adjoint problem
        sKET = (E[:,None,None] * KeET).flatten()
        # transpose of force due to heat expansion by temperature
        K_ET = coo_matrix((sKET,(iKET,jKET)),shape=(ndofT,ndofE)).tocsc()
        # Remove constrained dofs from matrix
        K_ET = K_ET[freeT,:][:,freeE]
        if solver == "lu":
            hT[freeT] = lu_T(K_ET@hE[freeE,:])
        elif solver == "direct":
            hT[freeT,0] = spsolve(K_T, K_ET@hE[freeE,0])
        # sensitivity
        dc[:]= penal*xPhys**(penal-1)*(\
                (kmax-kmin)*(np.dot(hT[edofMatT,0], KeT)*T[edofMatT,0]).sum(1)\
                +(Emax-Emin)*( np.dot(hE[edofMatE,0], KeE)*u[edofMatE,0] \
                               - hE[edofMatE,0]*fTe[:,:,0] ).sum(1))
        #
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
#element stiffness matrix for linear elasticity
def lkE(E,nu):
    """
    Create element stiffness matrix for 2D linear elasticity equation with 
    bilinear quadrilateral elements in plane stress. Taken from the 88 line code.
    
    Returns
    -------
    Ke : np.ndarray, shape (8,8)
        element stiffness matrix.
        
    """
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
#element stiffness matrix for heat conduction
def lkT():
    """
    Create element stiffness matrix for 2D Poisson equation with bilinear
    quadrilateral elements. Taken from the standard Sigmund textbook.
    
    Returns
    -------
    Ke : np.ndarray, shape (4,4)
        element stiffness matrix.
        
    """
    return np.array([[2/3, -1/6, -1/3, -1/6],
                     [-1/6, 2/3, -1/6, -1/3],
                     [-1/3, -1/6, 2/3, -1/6],
                     [-1/6, -1/3, -1/6, 2/3]])
# 
def lkET(E,nu,a):
    """
    Create force vector for 2D heat expansion with 
    bilinear quadrilateral Lagrangian elements. This amounts to
    
    int_Omega B_T @ C_v @ alpha_v @ N_T @ dOmega DeltaT
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    a : float 
        linear heat expansion coefficient.
        
    Returns
    -------
    Ke : np.ndarray, shape (8) or (8,4)
        element expansion matrix. returns nodal forces due to heat expansion by
        fT = Ke@Te where Te are the nodal temperatures.
        
    """
    return E * a/(3*(nu - 1)) * np.array([[1, 1, 1/2, 1/2],
                                          [1, 1/2, 1/2, 1],
                                          [-1, -1, -1/2, -1/2],
                                          [1/2, 1, 1, 1/2],
                                          [-1/2, -1/2, -1, -1],
                                          [-1/2, -1, -1, -1/2],
                                          [1/2, 1/2, 1, 1],
                                          [-1, -1/2, -1/2, -1]])
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
    move=0.05
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
