import numpy as np
from scipy.sparse import coo_matrix,coo_array
from scipy.sparse.linalg import spsolve,factorized
from scipy.optimize import minimize
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft,
         solver="lu"):
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
    print("Filter method: " + ["Sensitivity based","Density based","eta-projection"][ft])
    # Max and min stiffness
    E1=0.35
    E2=1.0
    # Poisson's ratio
    nu = 0.3
    #
    kappa1 = E1 / (2 * (1-nu))
    kappa2 = E2 / (2 * (1-nu))
    # heat expansion coefficients
    a1 = 1e-1
    a2 = 5e-2
    # stiffness constants springs
    kout = 0.
    # Allocate design variables (as array), initialize and allocate sens.
    x=np.ones(nely*nelx,dtype=float,order="F")
    xold=x.copy()
    xPhys=x.copy()
    g=0 # must be initialized to use the NGuyen/Paulino OC approach
    if ft == 2:
        beta = 1
        eta = 0.5
        xTilde = x.copy()
    # fetch element stiffness matrix
    KeE = lkE(E=1.0,nu=nu)
    KeET = lkET(E=1.0,nu=nu,a=1.0)
    # # dofs:
    ndofT = 4*(nelx+1)*(nely+1)
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
    iK = np.tile(edofMatE,KeE.shape[-1]).flatten()
    jK = np.repeat(edofMatE,KeE.shape[-1]).flatten()
    # assemble filter
    H,Hs = assemble_filter(rmin=rmin,el=el,nelx=nelx,nely=nely)
    # Solution, RHS
    T = np.ones((ndofT, 1))
    # BC's and support of linear elasticity
    dofs = np.arange(ndofE)
    fixed = np.union1d(dofs[0:2*(nely+1):2], # symmetry 
                       np.array([2*(nely+1)-1])) # bottom support
    dout = ndofE - 2 * (nely+1) + 2
    # Solution, RHS and adjoint vectors
    f = np.zeros((ndofE, 1))
    u = np.zeros((ndofE, 1))
    h = np.zeros((ndofE,1))
    # general
    free = np.setdiff1d(dofs, fixed)
    # indicator array for the output node and later for the adjoint problem
    l = np.zeros((ndofE, 1))
    l[np.arange(0,2*(nelx+1)*(nely+1),2*(nely+1))+1,0] = 1
    mask = l[:,0]!=0
    u0 = np.arange(0,nelx+1)**2 * 5e-3 + 3.0
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
    rhs_adj = np.zeros(l.shape) 
    while change>0.01 and loop<200:
        loop=loop+1 
        # Setup and solve elastic FE problem
        E = (E1+(xPhys)**penal*(E2-E1))
        #kappa = E / (2 * (1-nu))
        #a = (a1 * kappa1 * (kappa2 - kappa) - a2 * kappa2 * (kappa1 - kappa)) \
        #    / (kappa * (kappa1-kappa2))
        a = (a1+(xPhys)**penal*(a2-a1))
        sK=((KeE.flatten()[np.newaxis]).T*E).flatten(order='F')
        K_E = coo_array((sK,(iK,jK)),shape=(ndofE,ndofE)).tocsc()
        # add springs to stiffness matrix
        K_E[dout,dout] += kout
        # Remove constrained dofs from matrix
        K_E = K_E[free,:][:,free]
        # create right hand side
        fTe = KeET@T[edofMatT]
        # assemble
        fT = np.zeros(f.shape)
        np.add.at(fT[:,0],
                  edofMatE.flatten(),
                  (E[:,None,None] * a[:,None,None] * fTe).flatten())
        # Solve system 
        if solver == "lu":
            lu = factorized(K_E)
            u[free,0]=lu(f[free,0] + fT[free,0])
        elif solver == "direct":
            u[free,0]=spsolve(K_E,f[free,0] + fT[free,0])
        # Objective
        obj = ((u[mask] - u0)**2).mean()
        # adjoint problem
        rhs_adj[mask,0] = (-2)*(u[mask,0]-u0) / u0.shape[0] 
        if solver == "lu":
            h[free,0] = lu( rhs_adj[free,0] )
        elif solver == "direct":
            h[free,0] = spsolve(K_E,rhs_adj[free,0])
        # sensitivity
        dc[:]= penal*xPhys**(penal-1)*(\
                (E2-E1)*( np.dot(h[edofMatE,0], KeE)*u[edofMatE,0] \
                                 -E[:,None] * (a2-a1) *\
                                  h[edofMatE,0]*fTe[:,:,0] 
                                 -a[:,None] * h[edofMatE,0]*fTe[:,:,0]).sum(1))
        #kappa1*kappa2 / (kappa1-kappa2) *(a1-a2)/(kappa[:,None]**2) * 1/(2*(1-nu)) *\
        #
        dv[:] = np.ones(nely*nelx)
        # Sensitivity filtering:
        if ft==0:
            dc[:] = np.asarray((H*(x*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,x)
        elif ft==1:
            dc[:] = np.asarray(H*(dc[np.newaxis].T/Hs))[:,0]
            dv[:] = np.asarray(H*(dv[np.newaxis].T/Hs))[:,0]
        elif ft==2:
            dx = beta * (1 - np.tanh(beta * (xTilde - eta))**2) /\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
            dc[:] = np.asarray(H*((dc*dx)[np.newaxis].T/Hs))[:, 0] 
            dv[:] = np.asarray(H*((dv*dx)[np.newaxis]/Hs))[:, 0] 
        # Optimality criteria
        xold[:]=x
        x[:],g=oc(x,volfrac,dc,dv,g)
        # Filter design variables
        if ft==0:   
            xPhys[:]=x
        elif ft==1:    
            xPhys[:]=np.asarray(H*x[np.newaxis].T/Hs)[:,0]
        elif ft==2:
            xTilde = np.asarray(H*x[np.newaxis].T/Hs)[:, 0]
            # get volume preserving eta
            eta = find_eta(eta, xTilde, beta, volfrac)
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xTilde - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        # Compute the change by the inf. norm
        change=np.abs(x-xold).max()
        # Plot to screen
        im.set_array(-xPhys.reshape((nely,nelx),order="F"))
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(\
                    loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change))
        if (ft in [2]) and (beta < 16) and \
            (loop%50 == 0 or change < 0.01):
            beta = 2 * beta
            print(f"Parameter beta increased to {beta}")
    # Make sure the plot stays and that the shell remains    
    plt.show()
    input("Press any key...")
    #
    from topoptlab.output_designs import export_vtk, threshold
    export_vtk(filename="topoptbh_control",
               nelx=nelx,nely=nely,nelz=None,
               xPhys=xPhys,x=x,
               u=u,f=f,volfrac=volfrac)
    #
    xThresh = threshold(xPhys,
                        volfrac)
    # Setup and solve elastic FE problem
    E = (E1+(xThresh)**penal*(E2-E1))
    a = (a1+(xThresh)**penal*(a2-a1))
    sK=((KeE.flatten()[np.newaxis]).T*E).flatten(order='F')
    K_E = coo_matrix((sK,(iK,jK)),shape=(ndofE,ndofE)).tocsc()
    # add springs to stiffness matrix
    K_E[dout,dout] += kout
    # Remove constrained dofs from matrix
    K_E = K_E[free,:][:,free]
    # create right hand side
    fTe = KeET@T[edofMatT]
    # assemble
    fT = np.zeros(f.shape)
    np.add.at(fT[:,0],
              edofMatE.flatten(),
              (E[:,None,None] * a[:,None,None] * fTe).flatten())
    # Solve system 
    u_bw = np.zeros(u.shape)
    if solver == "lu":
        lu_E = factorized(K_E)
        u_bw[free,0]=lu_E(f[free,0] + fT[free,0])
    elif solver == "direct":
        u_bw[free,0]=spsolve(K_E,f[free,0] + fT[free,0])
    # Objective
    obj = ((u_bw[l!=0] - u0)**2).sum()
    #
    print("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}".format(\
                loop+1,obj,xThresh.mean()))
    #
    export_vtk(filename="topoptbh_control-bw",
               nelx=nelx,nely=nely,nelz=None,
               xPhys=xPhys,x=x,
               u=u_bw,f=f,volfrac=volfrac)
    #
    plt.close()
    fig,ax = plt.subplots(1,1)
    ax.plot(np.arange( int((l!=0).sum())),
            u0,
            label="target")
    ax.plot(np.arange( int((l!=0).sum())),
            u_bw[l!=0],
            label="gray design")
    ax.plot(np.arange( int((l!=0).sum())),
            u[l!=0],
            label="black/white design")
    ax.set_xlabel("position x")
    ax.set_ylabel("displacement")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
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
#element stiffness matrix 
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
    move=0.1
    damp = 0.3
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid=0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10,
                                                                    -dc/dv/lmid)**damp))))
        gt=xnew.mean()-volfrac
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return xnew,gt
def find_eta(eta0,xTilde,beta,volfrac):
    """
    Find volume preserving eta for the relaxed Haeviside projection similar to 
    what has been done in 
    
    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside 
        function which si recovered in the limit of beta to infinity
    rmin : float
        filter radius. 
    volfrac : float
        volume fraction. 

    Returns
    -------
    eta : float
        filter threshold.

    """
    result = minimize(_eta_residual, x0=eta0,
                      bounds=[[0., 1.]], options={"maxiter":1e5},
                      method='Nelder-Mead',jac=True,tol=1e-10,
                      args=(xTilde,beta,volfrac))
    if result.success:
        return result.x
    else:
        raise ValueError("volume conserving eta could not be found: ",result)
def _eta_residual(eta,xTilde,beta,volfrac):
    """
    Residual for finding the volume preserving eta for the relaxed Haeviside 
    projection.
    
    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside 
        function which si recovered in the limit of beta to infinity
    rmin : float
        filter radius. 
    volfrac : float
        volume fraction. 

    Returns
    -------
    residual : float
        residual of the volume constraint.

    """
    # Calculate the expression for given eta
    xPhys = (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
    #grad = -beta * np.sinh(beta)**(-1) * np.cosh(beta * (xTilde - eta))**(-2) * \
    #        np.sinh(xTilde * beta) * np.sinh((1 - xTilde) * beta)
    return (np.mean(xPhys) - volfrac)**2, None#, grad.mean()
# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=120
    nely=40
    volfrac=0.5
    rmin=2.4
    penal=3.0
    ft=1 # ft==0 -> sens, ft==1 -> dens
    import sys
    if len(sys.argv)>1: nelx   =int(sys.argv[1])
    if len(sys.argv)>2: nely   =int(sys.argv[2])
    if len(sys.argv)>3: volfrac=float(sys.argv[3])
    if len(sys.argv)>4: rmin   =float(sys.argv[4])
    if len(sys.argv)>5: penal  =float(sys.argv[5])
    if len(sys.argv)>6: ft     =int(sys.argv[6])
    main(nelx,nely,volfrac,penal,rmin,ft)
