import numpy as np
from scipy.sparse import csc_array

def find_eta(eta,xTilde,beta,volfrac):
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
        """
        print("x")
        print(x)
        print("Xi")
        print(Xi)
        print("xi")
        print(xi)
        print("dfxi")
        print(dfxi)
        print("dfx")
        print(dfx)
        """
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
            """
            print("cbr")
            print(cbr)
            print("qs")
            print(qs)
            """
            dsmaxdxi = csc_array((qs[1:-1],(qi[1:-1], qj[1:-1])), 
                                   shape=(nelx, nelx))
            """
            print("lambda before update")    
            print(lambda_vals)
            print("dsmindx")
            print(dsmindx)
            """
            # Update sensitivities
            for k in np.arange(nSens):
                dfx[i,:,k] = dsmindx * (dfxi[i,:,k] + lambda_vals[:,k])
                lambda_vals[:,k] = ((dfxi[i,:,k] + lambda_vals[:,k]) * dsmindXi) @ dsmaxdxi
            """
            print("dfx")
            print(dfx[i,:,k])
            print("lambda")    
            print(lambda_vals[:,k])
            """
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

# The real main driver
if __name__ == "__main__":
    # Default input parameters
    nelx = 60  # 180
    nely = int(nelx/3)  # 60
    volfrac = 0.5  # 0.4
    rmin = 0.04*nelx  # 5.4
    penal = 3.0
    ft = 6 # ft==0 -> sens, ft==1 -> dens
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
    try:
        main(nelx, nely, volfrac, penal, rmin, ft,
             manufact = None,
             passive=False,pde=False,solver="oc",
             nouteriter=2000,
             ninneriter=0,
             debug=False)
    except Exception as e:
        logging.error(traceback.format_exc())
        logging.shutdown()
