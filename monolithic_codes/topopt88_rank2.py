# A 165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN, JANUARY 2013
# minor modifications by Stefan Hiemer (January 2025)
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# MAIN DRIVER
def main(nelx,nely,volfrac,penal,rmin,ft,
         angle_update="principle_stress"):
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
    # Max and min stiffness
    E0=1e-1
    E=1.0
    nu = 0.3
    # isotropic background stiffness tensor
    c0 = E0 / (1-nu**2) * np.array([[1,nu,0], 
                                    [nu,1,0],
                                    [0,0,(1-nu)/2]])[None,:,:]
    # Allocate design variables
    # layer widths
    mu = np.full( shape=(nelx*nely,2),
                  fill_value=1- np.sqrt(1-volfrac))
    muPhys = mu.copy()
    muold = mu.copy()
    rho = muPhys[:,0]+muPhys[:,1]-muPhys[:,0]*muPhys[:,1]
    # angles 
    angs = np.zeros((nelx*nely,1))
    angsold = angs.copy() 
    g=0
    # fetch element stiffness matrix
    KE = lk(c0)
    # fetch bmatrix for midpoint of element 
    bmatr = np.array([[-0.25,  0.  ,  0.25,  0.  ,  0.25,  0.  , -0.25,  0.  ],
                      [ 0.  , -0.25,  0.  , -0.25,  0.  ,  0.25,  0.  ,  0.25],
                      [-0.25, -0.25, -0.25,  0.25,  0.25,  0.25,  0.25, -0.25]])
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
    dofs=np.arange(2*(nelx+1)*(nely+1))
    fixed = np.hstack((np.arange(0,2*(nely+1),2), # symmetry 
                       np.array([2*(nelx+1)*(nely+1)-1]))) # fixation bottom right
    free=np.setdiff1d(dofs,fixed)
    # Solution and RHS vectors
    f=np.zeros((ndof,1))
    u=np.zeros((ndof,1))
    # Set load
    f[1,0]=-1
    # Initialize plot and plot the initial design
    plt.ion() # Ensure that redrawing is possible
    fig,ax = plt.subplots()
    im = ax.imshow(-rho.reshape((nely,nelx),order="F"), cmap='gray',
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
    dobjdmu1 = np.ones(nely*nelx)
    dobjdmu2 = np.ones(nely*nelx)
    dvdmu1 = np.ones(nely*nelx)
    dvdmu2 = np.ones(nely*nelx)
    ce = np.ones(nely*nelx)
    stress = np.zeros(u.shape)
    while change>0.01 and loop<2000:
        loop=loop+1
        # update local properties
        R = Rv_2d(angs)
        cH = rank2_2d(muPhys[:,0], muPhys[:,1], nu=nu, E=E)
        c = R.transpose((0,2,1))@cH@R
        # add background stiffness
        c = c + c0
        # Setup and solve FE problem
        lks = lk(c) # element stiffness matrices
        K = coo_matrix((lks.flatten(),(iK,jK)),shape=(ndof,ndof)).tocsc()
        # Remove constrained dofs from matrix
        K = K[free,:][:,free]
        print(K.todense())
        import sys 
        sys.exit()
        # Solve system 
        u[free,0]=spsolve(K,f[free,0])
        #print(K.todense())
        #print(u)
        # update angles based on principal stress:
        angsold[:] = angs
        if angle_update == "principle_stress":
            # calculate strain and stress
            strain = bmatr@u[edofMat]
            stress = c@strain
            #
            angs[:,0] =  np.atan2(2*stress[:,2,0] , 
                                  stress[:,0,0] - stress[:,1,0])
        change=np.abs(angs-angsold).max()
        # objective
        ce[:] = np.squeeze(u[edofMat].transpose((0,2,1)) @ lks @ u[edofMat])
        obj= ce.sum()
        # sensitivity
        dcdmu1 = rank2_2d_dmu1(mu1=mu[:,0], mu2=mu[:,1], nu=nu, E=E)
        dcdmu1[:] = R.transpose((0,2,1))@dcdmu1@R
        dobjdmu1[:]= -np.squeeze(u[edofMat].transpose((0,2,1)) @ lk(c=dcdmu1) @ u[edofMat])
        #
        dcdmu2 = rank2_2d_dmu2(mu1=mu[:,0], mu2=mu[:,1], nu=nu, E=E)
        dcdmu2[:] = R.transpose((0,2,1))@dcdmu2@R
        dobjdmu2[:]= -np.squeeze(u[edofMat].transpose((0,2,1)) @ lk(c=dcdmu2) @ u[edofMat])
        #
        dmu = 1e-10
        _dobjdmu2 = np.zeros(dobjdmu2.shape)
        for i in np.arange(nelx*nely):
            _u = np.zeros(u.shape)
            _ce = np.zeros(ce.shape)
            #
            _mu = mu.copy()
            _mu[i,1] = _mu[i,1] + dmu 
            #
            cH = rank2_2d(_mu[:,0], _mu[:,1], nu=nu, E=E)
            c = R.transpose((0,2,1))@cH@R
            c = c + c0
            #
            _lks = lk(c) # element stiffness matrices
            _K = coo_matrix((_lks.flatten(),(iK,jK)),shape=(ndof,ndof)).tocsc()
            # Remove constrained dofs from matrix
            _K = _K[free,:][:,free]
            # Solve system 
            _u[free,0]=spsolve(_K,f[free,0])
            #
            _ce[:] = np.squeeze(_u[edofMat].transpose((0,2,1)) @ _lks @ _u[edofMat])
            _obj= _ce.sum()
            #
            _dobjdmu2[i] = (_obj-obj)/dmu
        print(dobjdmu2)
        print(_dobjdmu2)
        import sys 
        sys.exit()
        #
        dvdmu1[:] = 1-muPhys[:,1]
        dvdmu2[:] = 1-muPhys[:,0]
        # Sensitivity filtering:
        if ft==0:
            dobjdmu1[:] = np.asarray((H*(mu[:,0]*dobjdmu1))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,mu[:,0])
            dobjdmu2[:] = np.asarray((H*(mu[:,1]*dobjdmu2))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,mu[:,1])
        elif ft==1:
            dobjdmu1[:] = np.asarray(H*(dobjdmu1[np.newaxis].T/Hs))[:,0]
            dobjdmu2[:] = np.asarray(H*(dobjdmu2[np.newaxis].T/Hs))[:,0]
            dvdmu1[:] = np.asarray(H*(dvdmu1[np.newaxis].T/Hs))[:,0]
            dvdmu2[:] = np.asarray(H*(dvdmu2[np.newaxis].T/Hs))[:,0]
        # Optimality criteria
        muold[:]=mu
        mu[:],g=oc(mu,volfrac,dobjdmu1,dobjdmu2,dvdmu1,dvdmu2,g)
        # Filter design variables
        if ft==0:   
            muPhys[:]=mu
        elif ft==1:    
            muPhys[:,0]=np.asarray(H*mu[:,0][np.newaxis].T/Hs)[:,0]
            muPhys[:,1]=np.asarray(H*mu[:,1][np.newaxis].T/Hs)[:,0]
        # Compute the change by the inf. norm
        change=np.abs(mu-muold).max()
        # Plot to screen
        rho = muPhys[:,0]+muPhys[:,1]-muPhys[:,0]*muPhys[:,1]
        im.set_array(-rho.reshape((nely,nelx),order="F"))
        fig.canvas.draw()
        plt.pause(0.01)
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , obj.: {1:.10f} Vol.: {2:.10f}, ch.: {3:.10f}".format(\
                    loop,obj,rho.mean(),change))
        #if loop%50:
        #    c0 = c0 /10
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
def Rv_2d(theta):
    """
    2D rotation matrix for tensors of 2nd order ("Voigt vectors") and 4th order 
    ("Voigt matrices"). 

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,3,3)
        rotation matrices.

    """
    
    return np.column_stack((np.cos(theta)**2, np.sin(theta)**2, -np.sin(2*theta)/2, 
                            np.sin(theta)**2, np.cos(theta)**2, np.sin(2*theta)/2, 
                            np.sin(2*theta), -np.sin(2*theta), np.cos(2*theta)))\
          .reshape((theta.shape[0],3,3))
#element stiffness matrix
def lk(c):
    """
    Create element stiffness matrix for 2D anisotropic linear elasticity with 
    bilinear quadrilateral elements. 
    
    Parameters
    ----------
    c : np.ndarray, shape (nel,3,3)
        stiffness tensor.
    
    Returns
    -------
    Ke : np.ndarray, shape (nel,8,8)
        element stiffness matrix.
        
    """
    Ke = np.column_stack((c[:,0,0]/3 + c[:,0,2]/4 + c[:,2,0]/4 + c[:,2,2]/3, 
                          c[:,0,1]/4 + c[:,0,2]/3 + c[:,2,1]/3 + c[:,2,2]/4, 
                          -c[:,0,0]/3 + c[:,0,2]/4 - c[:,2,0]/4 + c[:,2,2]/6, 
                          c[:,0,1]/4 - c[:,0,2]/3 + c[:,2,1]/6 - c[:,2,2]/4, 
                          -c[:,0,0]/6 - c[:,0,2]/4 - c[:,2,0]/4 - c[:,2,2]/6, 
                          -c[:,0,1]/4 - c[:,0,2]/6 - c[:,2,1]/6 - c[:,2,2]/4, 
                          c[:,0,0]/6 - c[:,0,2]/4 + c[:,2,0]/4 - c[:,2,2]/3, 
                          -c[:,0,1]/4 + c[:,0,2]/6 - c[:,2,1]/3 + c[:,2,2]/4, 
                          c[:,1,0]/4 + c[:,1,2]/3 + c[:,2,0]/3 + c[:,2,2]/4, 
                          c[:,1,1]/3 + c[:,1,2]/4 + c[:,2,1]/4 + c[:,2,2]/3, 
                          -c[:,1,0]/4 + c[:,1,2]/6 - c[:,2,0]/3 + c[:,2,2]/4, 
                          c[:,1,1]/6 - c[:,1,2]/4 + c[:,2,1]/4 - c[:,2,2]/3, 
                          -c[:,1,0]/4 - c[:,1,2]/6 - c[:,2,0]/6 - c[:,2,2]/4, 
                          -c[:,1,1]/6 - c[:,1,2]/4 - c[:,2,1]/4 - c[:,2,2]/6, 
                          c[:,1,0]/4 - c[:,1,2]/3 + c[:,2,0]/6 - c[:,2,2]/4, 
                          -c[:,1,1]/3 + c[:,1,2]/4 - c[:,2,1]/4 + c[:,2,2]/6, 
                          -c[:,0,0]/3 - c[:,0,2]/4 + c[:,2,0]/4 + c[:,2,2]/6, 
                          -c[:,0,1]/4 - c[:,0,2]/3 + c[:,2,1]/6 + c[:,2,2]/4, 
                          c[:,0,0]/3 - c[:,0,2]/4 - c[:,2,0]/4 + c[:,2,2]/3, 
                          -c[:,0,1]/4 + c[:,0,2]/3 + c[:,2,1]/3 - c[:,2,2]/4, 
                          c[:,0,0]/6 + c[:,0,2]/4 - c[:,2,0]/4 - c[:,2,2]/3, 
                          c[:,0,1]/4 + c[:,0,2]/6 - c[:,2,1]/3 - c[:,2,2]/4, 
                          -c[:,0,0]/6 + c[:,0,2]/4 + c[:,2,0]/4 - c[:,2,2]/6, 
                          c[:,0,1]/4 - c[:,0,2]/6 - c[:,2,1]/6 + c[:,2,2]/4, 
                          c[:,1,0]/4 + c[:,1,2]/6 - c[:,2,0]/3 - c[:,2,2]/4, 
                          c[:,1,1]/6 + c[:,1,2]/4 - c[:,2,1]/4 - c[:,2,2]/3, 
                          -c[:,1,0]/4 + c[:,1,2]/3 + c[:,2,0]/3 - c[:,2,2]/4, 
                          c[:,1,1]/3 - c[:,1,2]/4 - c[:,2,1]/4 + c[:,2,2]/3, 
                          -c[:,1,0]/4 - c[:,1,2]/3 + c[:,2,0]/6 + c[:,2,2]/4, 
                          -c[:,1,1]/3 - c[:,1,2]/4 + c[:,2,1]/4 + c[:,2,2]/6, 
                          c[:,1,0]/4 - c[:,1,2]/6 - c[:,2,0]/6 + c[:,2,2]/4, 
                          -c[:,1,1]/6 + c[:,1,2]/4 + c[:,2,1]/4 - c[:,2,2]/6, 
                          -c[:,0,0]/6 - c[:,0,2]/4 - c[:,2,0]/4 - c[:,2,2]/6, 
                          -c[:,0,1]/4 - c[:,0,2]/6 - c[:,2,1]/6 - c[:,2,2]/4, 
                          c[:,0,0]/6 - c[:,0,2]/4 + c[:,2,0]/4 - c[:,2,2]/3, 
                          -c[:,0,1]/4 + c[:,0,2]/6 - c[:,2,1]/3 + c[:,2,2]/4, 
                          c[:,0,0]/3 + c[:,0,2]/4 + c[:,2,0]/4 + c[:,2,2]/3, 
                          c[:,0,1]/4 + c[:,0,2]/3 + c[:,2,1]/3 + c[:,2,2]/4, 
                          -c[:,0,0]/3 + c[:,0,2]/4 - c[:,2,0]/4 + c[:,2,2]/6, 
                          c[:,0,1]/4 - c[:,0,2]/3 + c[:,2,1]/6 - c[:,2,2]/4, 
                          -c[:,1,0]/4 - c[:,1,2]/6 - c[:,2,0]/6 - c[:,2,2]/4, 
                          -c[:,1,1]/6 - c[:,1,2]/4 - c[:,2,1]/4 - c[:,2,2]/6, 
                          c[:,1,0]/4 - c[:,1,2]/3 + c[:,2,0]/6 - c[:,2,2]/4, 
                          -c[:,1,1]/3 + c[:,1,2]/4 - c[:,2,1]/4 + c[:,2,2]/6, 
                          c[:,1,0]/4 + c[:,1,2]/3 + c[:,2,0]/3 + c[:,2,2]/4, 
                          c[:,1,1]/3 + c[:,1,2]/4 + c[:,2,1]/4 + c[:,2,2]/3, 
                          -c[:,1,0]/4 + c[:,1,2]/6 - c[:,2,0]/3 + c[:,2,2]/4, 
                          c[:,1,1]/6 - c[:,1,2]/4 + c[:,2,1]/4 - c[:,2,2]/3, 
                          c[:,0,0]/6 + c[:,0,2]/4 - c[:,2,0]/4 - c[:,2,2]/3, 
                          c[:,0,1]/4 + c[:,0,2]/6 - c[:,2,1]/3 - c[:,2,2]/4, 
                          -c[:,0,0]/6 + c[:,0,2]/4 + c[:,2,0]/4 - c[:,2,2]/6, 
                          c[:,0,1]/4 - c[:,0,2]/6 - c[:,2,1]/6 + c[:,2,2]/4, 
                          -c[:,0,0]/3 - c[:,0,2]/4 + c[:,2,0]/4 + c[:,2,2]/6, 
                          -c[:,0,1]/4 - c[:,0,2]/3 + c[:,2,1]/6 + c[:,2,2]/4, 
                          c[:,0,0]/3 - c[:,0,2]/4 - c[:,2,0]/4 + c[:,2,2]/3, 
                          -c[:,0,1]/4 + c[:,0,2]/3 + c[:,2,1]/3 - c[:,2,2]/4, 
                          -c[:,1,0]/4 - c[:,1,2]/3 + c[:,2,0]/6 + c[:,2,2]/4, 
                          -c[:,1,1]/3 - c[:,1,2]/4 + c[:,2,1]/4 + c[:,2,2]/6, 
                          c[:,1,0]/4 - c[:,1,2]/6 - c[:,2,0]/6 + c[:,2,2]/4, 
                          -c[:,1,1]/6 + c[:,1,2]/4 + c[:,2,1]/4 - c[:,2,2]/6, 
                          c[:,1,0]/4 + c[:,1,2]/6 - c[:,2,0]/3 - c[:,2,2]/4, 
                          c[:,1,1]/6 + c[:,1,2]/4 - c[:,2,1]/4 - c[:,2,2]/3, 
                          -c[:,1,0]/4 + c[:,1,2]/3 + c[:,2,0]/3 - c[:,2,2]/4, 
                          c[:,1,1]/3 - c[:,1,2]/4 - c[:,2,1]/4 + c[:,2,2]/3))
    return Ke.reshape(c.shape[0],8,8)
def rank2_2d(mu1,mu2,nu,E):
    """
    Stiffness tensor for rank2 laminate in 2D taken from 
    "Wu, Jun, Ole Sigmund, and Jeroen P. Groen. "Topology optimization of multi-scale structures: a review." Structural and Multidisciplinary Optimization 63 (2021): 1455-1480.".
    
    Parameters
    ----------
    mu1 : np.ndarray, shape (nel)
        relative layer widths of first (smaller) layer.
    mu2 : np.ndarray, shape (nel)
        relative layer widths of second (bigger) layer.
    nu : float
        Poisson's ratio of base material.
    E : float
        Young's modulus of base material.
    
    Returns
    -------
    c : np.ndarray, shape (nel,3,3)
        stiffness tensor.
    """
    factor = E/(1-mu2+((mu1*mu2)*(1-nu)))
    _0 = np.zeros(mu1.shape)
    return factor[:,None,None] * np.column_stack((mu1,mu1*mu2*nu,_0,
                                                  mu1*mu2*nu,mu2*(1-mu2+(mu1*mu2)),_0,
                                                  _0,_0,_0)).reshape(mu1.shape[0],3,3)
def rank2_2d_dmu1(mu1,mu2,nu,E):
    """
    Derivative of stiffness tensor for rank2 laminate in 2D for the layer width
    mu1.
    
    Parameters
    ----------
    mu1 : np.ndarray, shape (nel)
        relative layer widths of first (smaller) layer.
    mu2 : np.ndarray, shape (nel)
        relative layer widths of second (bigger) layer.
    nu : float
        Poisson's ratio of base material.
    E : float
        Young's modulus of base material.
    
    Returns
    -------
    dcdmu1 : np.ndarray, shape (nel,3,3)
        stiffness tensor.
    """
    factor = E/(1-mu2+mu1*mu2*(1-nu))
    factordmu1 = (-1) * E / (1-mu2+mu1*mu2*(1-nu))**2 * (mu2*(1-nu))
    _0 = np.zeros(mu1.shape)
    A = np.column_stack((mu1, mu1*mu2*nu, _0,
                         mu1*mu2*nu, mu2*(1-mu2+mu1*mu2), _0,
                         _0, _0, _0)).reshape(mu1.shape[0],3,3)
    Admu1 = np.column_stack((np.ones(mu1.shape),mu2*nu,_0,
                             mu2*nu,mu2**2,_0,
                             _0,_0,_0)).reshape(mu1.shape[0],3,3)
    return factordmu1[:,None,None] * A + factor[:,None,None] * Admu1 
def rank2_2d_dmu2(mu1,mu2,nu,E):
    """
    Derivative of stiffness tensor for rank2 laminate in 2D for the layer width
    mu2.
    
    Parameters
    ----------
    mu1 : np.ndarray, shape (nel)
        relative layer widths in first (smaller) layer.
    mu2 : np.ndarray, shape (nel)
        relative layer widths of second (bigger) layer.
    nu : float
        Poisson's ratio of base material.
    E : float
        Young's modulus of base material.
    
    Returns
    -------
    dcdmu2 : np.ndarray, shape (nel,3,3)
        stiffness tensor.
    """
    factor = E/(1-mu2+mu1*mu2*(1-nu))
    factordmu2 = (-E) / (1-mu2+mu1*mu2*(1-nu))**2 * (mu1*(1-nu)-1)
    _0 = np.zeros(mu1.shape)
    A = np.column_stack((mu1, mu1*mu2*nu, _0,
                         mu1*mu2*nu, mu2*(1-mu2+mu1*mu2), _0,
                         _0, _0, _0)).reshape(mu1.shape[0],3,3)
    Admu2 = np.column_stack((_0, mu1*nu, _0,
                             mu1*nu, (1-2*mu2+2*mu1*mu2), _0,
                             _0, _0, _0)).reshape(mu1.shape[0],3,3)
    return factordmu2[:,None,None] * A + factor[:,None,None] * Admu2 
# Optimality criterion
def oc(mu,volfrac,dobjdmu1,dobjdmu2,dvdmu1,dvdmu2,g):
    """
    Optimality criteria method (section 2.2 in top88 paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Overtaken and adapted from the 
    
    165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN
    
    Parameters
    ----------
    mu : np.array, shape (nel)
        relative layer widths for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dobjdmu1 : np.array, shape (nel)
        gradient of objective function/complicance with respect to  
        smaller relative layer width.
    dobjdmu2 : np.array, shape (nel)
        gradient of objective function/complicance with respect to  
        second (bigger) relative layer width.
    dvdmu1 : np.array, shape (nel)
        gradient of volume constrain with respect to first (smaller) relative 
        layer width.
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
    l2=1e5
    move=0.05
    # reshape to perform vector operations
    munew=np.zeros(mu.shape)
    if (dobjdmu1>=0).any() or (dobjdmu2>=0).any():
        raise ValueError("Still something wrong")
    while (l2-l1)/(l1+l2)>1e-3:
        lmid=0.5*(l2+l1)
        munew[:,0]= np.maximum(0.,
                            np.maximum(mu[:,0]-move,
                                       np.minimum(1.,
                                                  np.minimum(mu[:,0]+move,
                                                             mu[:,0]*np.sqrt(-dobjdmu1/dvdmu1/lmid)))))
        munew[:,1]= np.maximum(0.,
                            np.maximum(mu[:,1]-move,
                                       np.minimum(1.,
                                                  np.minimum(mu[:,1]+move,
                                                             mu[:,1]*np.sqrt(-dobjdmu2/dvdmu2/lmid)))))
        rho = munew[:,0]+munew[:,1] - (munew[:,0]*munew[:,1])
        gt=(rho-volfrac).mean()
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return munew,gt
# The real main driver    
if __name__ == "__main__":
    # Default input parameters
    nelx=1
    nely=1
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
