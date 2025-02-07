import numpy as np
from scipy.sparse import csc_array,coo_matrix
from scipy.optimize import minimize
from scipy.ndimage import convolve

from topoptlab.elements.screenedpoisson_2d import lk_screened_poisson_2d

def assemble_matrix_filter(nelx,nely,rmin,el = None,
                           ndim=2):
    """
    Assemble distance based filters as sparse matrix that is applied on to
    densities/sensitivities by standard multiplication.
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    volfrac : float
        volume fraction. 
    rmin : float
        cutoff radius for the filter. Only elements within the element-center 
        to element center distance are used for filtering. 
    el : np.ndarray or None
        sorted array of element indices.
    ndim : int 
        number of dimensions

    Returns
    -------
    H : csc matrix
        unnormalized filter.
    Hs : np matrix
        normalization factor.

    """
    
    if el is None:
        el = np.arange(nelx*nely)
    
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
    # hat function
    fac = rmin-np.sqrt(((i-k)**2+(j-l)**2))
    iH[cc] = el # row
    jH[cc] = k*nely+l #column
    sH[cc] = np.maximum(0.0, fac)
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)
    return H,Hs

def assemble_convolution_filter(nelx,nely,rmin,
                                nelz=None,ndim=2):
    # filter radius in number of elements
    nfilter = int(2*np.floor(rmin)+1)
    #
    x = np.arange(-np.floor(rmin),rmin)
    #x = np.tile(x,tuple([nfilter])*(ndim-1) + tuple([1]))
    #
    
    if ndim == 2:
        x = np.tile(x,(nfilter,1))
        y = np.rot90(x)
        # hat function
        kernel = np.maximum(0.0,rmin - np.sqrt(x**2 + y**2))
        # normalization constant
        hs = convolve(np.ones((nelx, nely)).T,
                      kernel,
                      mode="constant",
                      cval=0).T.flatten()
        return kernel,hs
    else:
        raise NotImplementedError("3D not yet implemented")

def assemble_helmholtz_filter(nelx,nely,rmin,ndim=2,
                              el=None,n1=None,n2=None):
    """
    Assemble Helmholtz PDE based filter from "Efficient topology optimization 
    in MATLAB using 88 lines of code".
    
    This filter works by mappinging the element densities to nodes via the 
    operator TF, then performing the actual filter operation by solving a 
    Helmholtz / screened Poisson PDE in the standard FEM style with subsequent 
    back-mapping of the nodal (filtered) densities to the elements. 
    
    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    volfrac : float
        volume fraction. 
    rmin : float
        cutoff radius for the filter. Only elements within the element-center 
        to element center distance are used for filtering. 
    ndim : int 
        number of dimensions
    el : np.ndarray or None
        sorted array of element indices.
    n1 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    n2 : np.ndarray or None
        index array to help constructing the stiffness matrix.

    Returns
    -------
    KF : csc matrix
        stiffness matrix.
    TF : csc matrix
        mapping (or in this special case averaging) operator that maps element
        densities to nodes and its inverse maps nodal densities back to the 
        elements.

    """
    if ndim != 2:
        raise NotImplementedError()
    # element indices
    if el is None:
        el = np.arange(nelx*nely)
    # 
    if n1 is None:
        elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
        n1 = ((nely+1)*elx+ely).flatten()
        n2 = ((nely+1)*(elx+1)+ely).flatten()
    # conversion of filter radius via 1D Green's function
    Rmin = rmin/(2*np.sqrt(3))
    #
    KEF = lk_screened_poisson_2d(k=Rmin**2)
    ndofF = (nelx+1)*(nely+1)
    edofMatF = np.column_stack((n1, n2, n2 +1, n1 +1 ))
    iKF = np.kron(edofMatF, np.ones((4, 1))).flatten()
    jKF = np.kron(edofMatF, np.ones((1, 4))).flatten()
    sKF = np.tile(KEF.flatten(),nelx*nely)
    KF = coo_matrix((sKF, (iKF, jKF)), shape=(ndofF, ndofF)).tocsc()
    iTF = edofMatF.flatten(order='F')
    jTF = np.tile(el, 4)
    sTF = np.full(4*nelx*nely,1/4)
    TF = coo_matrix((sTF, (iTF, jTF)), shape=(ndofF,nelx*nely)).tocsc()
    return KF,TF

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
        unnormalized filter.

    """
    result = minimize(_eta_residual, x0=eta0,
                      bounds=[[0., 1.]], 
                      method='Nelder-Mead',jac=True,tol=1e-10,
                      args=(xTilde,beta,volfrac))
    if result.success:
        return result.x
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _eta_residual(eta,xTilde,beta,volfrac):
    """
    Residual for findined the volume preserving eta for the relaxed Haeviside 
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
    eta : float
        unnormalized filter.

    """
    # Calculate the expression for given eta
    xPhys = (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
    #grad = -beta * np.sinh(beta)**(-1) * np.cosh(beta * (xTilde - eta))**(-2) * \
    #        np.sinh(xTilde * beta) * np.sinh((1 - xTilde) * beta)
    return np.abs(np.mean(xPhys) - volfrac)**2, None#, grad.mean()

def AMfilter(x, baseplate='S', sensitivities=None):
    """
    Applies the filter by 
    
    Langelaar, Matthijs. "An additive manufacturing filter for topology optimization of print-ready designs." Structural and multidisciplinary optimization 55 (2017): 871-883.
    
    
    Applies a filter to densities that enforces that each density cannot be 
    larger then the maximum density of its supporting region.
    
    Parameters
    ----------
    x : np.ndarray
        Blueprint design (2D array), with values between 0 and 1 and shape 
        (nely,nelx). The shape is needed to determine the positions of elements
        with respect to the baseplate.
    baseplate : str, optional
        Character indicating baseplate orientation: 'N', 'E', 'S', 'W'. Default is 'S'.
        For 'X', the filter bypasses and returns the input as-is.
    sensitivities : np.ndarray
        sensitivities associated with the design input shape 
        (nely,nelx,nsens).
        
    Returns
    -------
    xi or sensitivities: np.ndarray
        Printable design density (nely,nelx) or sensitivities (nely,nelx,nSens) 
        after filtering.
    """
    # number of supporting elements
    Ns = 3
    # Constants for smooth max/min functions
    P,ep,xi_0 = 40,1e-4,.5
    Q = P + np.log(Ns) / np.log(xi_0)
    SHIFT = 100 * np.finfo(float).tiny**(1 / P)
    BACKSHIFT = 0.95 * Ns**(1 / Q) * SHIFT**(P / Q)
    # Check for bypass option
    if baseplate == 'X':
        return x, sensitivities  # Return as-is
    # Determine rotation based on baseplate orientation
    nRot = 'SWNE'.find(baseplate.upper())
    x = np.rot90(x, nRot).copy()
    # Initialize xi
    xi = np.zeros_like(x)
    nely, nelx = x.shape
    # loop for applying AM filter from top moving layer-wise downwards
    xi[-1, :] = x[-1,:].copy()  # Copy base row as-is
    Xi, keep,sq = [np.zeros_like(x) for i in np.arange(3)]
    #cbr = np.pad(xi[-1:0:-1,:],
    #             pad_width=((0,0),(1, 1)),
    #             mode='constant',
    #             constant_values=0) + SHIFT
    #print("cbr:\n",cbr)
    #print()
    for i in np.arange(nely-2,-1,-1):
        cbr = np.pad(xi[i+1,:] + SHIFT, 
                     (1, 1), 
                     'constant',
                     constant_values=SHIFT)
        #print(cbr)
        keep[i,:] = (cbr[:-2]**P + cbr[1:-1]**P + cbr[2:]**P)
        Xi[i,:] = keep[i,:]**(1 / Q) - BACKSHIFT
        sq[i,:] = np.sqrt((x[i,:] - Xi[i,:])**2 + ep)
        xi[i,:] = 0.5 * ((x[i,:] + Xi[i,:]) - sq[i,:] + np.sqrt(ep))
    # Process sensitivities if provided. 
    if sensitivities is not None:
        # 
        nSens = sensitivities.shape[-1]
        # sensitivities as obtained by the usual adjoint analysis. this must be
        # rotated and filtered
        dfxi = np.rot90(sensitivities, nRot)
        # filtered gradients/sensitivities
        dfx = np.zeros_like(dfxi)
        # precalculate indices later for fast multiplication via sparse matrix
        qi = np.repeat(np.arange(nelx), Ns)
        qj = np.tile([-1, 0, 1], nelx) + qi
        # Lagrangian multipliers for adjoint sensitivity analysis 
        lambda_vals = np.zeros((nelx,nSens))
        # iterate from top to base layer
        for i in np.arange(nely-1):
            # smin sensitivity terms
            dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
            dsmindXi = 1 - dsmindx
            # smax sensitivity terms
            cbr = np.pad(xi[i + 1, :] + SHIFT, 
                         (1, 1), 
                         'constant',
                         constant_values=SHIFT)  # Pad with zeros
            dmx = np.zeros((nelx,Ns))
            for j in np.arange(Ns):
                dmx[:,j] = (P/Q) * keep[i, :]**((1/Q) - 1) * cbr[np.arange(nelx)+j]**(P - 1)
            # Rearrange data for quick multiplication
            qs = dmx.flatten()
            dsmaxdxi = csc_array((qs[1:-1],(qi[1:-1], qj[1:-1])), 
                                   shape=(nelx, nelx))
            # Update sensitivities
            for k in np.arange(nSens):
                dfx[i,:,k] = dsmindx * (dfxi[i,:,k] + lambda_vals[:,k])
                lambda_vals[:,k] = ((dfxi[i,:,k] + lambda_vals[:,k]) * dsmindXi) @ dsmaxdxi
        # base layer 
        dfx[-1,:,:] = dfxi[-1,:,:]+lambda_vals[:,:]
    if sensitivities is None:
        # Rotate xi back to original orientation if rotated
        return np.rot90(xi, -nRot)
    else:
        # Rotate sensitivities back to original orientation if rotated
        #print("gradient before backrotation")
        #print(dfx) 
        return np.rot90(dfx, -nRot)