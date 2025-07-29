import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import root_scalar

from topoptlab.geometries import diracdelta
from topoptlab.utils import map_eltoimg,map_eltovoxel

from matplotlib.pyplot import subplots,figure,figaspect,show 

def find_eta(eta0, xTilde, beta, volfrac,
             **kwargs):
    """
    Find volume preserving eta for the relaxed Haeviside projection similar to
    what has been done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    # unfortunately scipy.optimize needs f to change sign between the
    # respective ends of the brackets, therefor the eta found by this function
    # is offset by -1/2 to the value later used
    result = root_scalar(f=_root_func,fprime=True,method="newton",
                         x0=eta0-1/2, x1=0., maxiter=1000,
                         args=(xTilde,beta,volfrac),
                         bracket=[-1/2,1/2])
    #
    if result.converged:
        return result.root+1/2
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _root_func(eta,xTilde,beta,volfrac):
    """
    Function whose root is the volume preserving threshold.

    Parameters
    ----------
    eta : float
        current threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.

    Returns
    -------
    res : float
        value of current volume fraction - intended volume fraction.
    gradient : float
        gradient for Newton procedure

    """
    #
    eta = eta + 1/2
    #
    xPhys = eta_projection(eta=eta,xTilde=xTilde,beta=beta)
    # terms
    tanh_bn = np.tanh(beta * eta)
    tanh_b1n = np.tanh(beta * (1 - eta))
    tanh_bx_n = np.tanh(beta * (xTilde - eta))
    tanh_bn_x = np.tanh(beta * (eta - xTilde))

    sech2_bn = 1 - tanh_bn**2
    sech2_bx_n = 1 - tanh_bx_n**2
    sech2_b1n = 1 - tanh_b1n**2
    #
    denom1 = tanh_bn + tanh_b1n
    denom2 = denom1 ** 2
    #
    term1 = -sech2_bx_n
    term2 =  sech2_bn * (tanh_b1n + tanh_bn_x) + sech2_b1n * (tanh_bn + tanh_bx_n)
    return xPhys.mean()-volfrac, beta*(term1/denom1 + term2/denom2).mean()

def eta_projection(eta,xTilde,beta):
    """
    eta projection.

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    return (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def AMfilter(x, baseplate='S', sensitivities=None):
    """
    Applies the filter by

    Langelaar, Matthijs. "An additive manufacturing filter for topology
    optimization of print-ready designs." Structural and multidisciplinary
    optimization 55 (2017): 871-883.


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
    # constants for smooth max/min functions
    P,ep,xi_0 = 40,1e-4,.5
    Q = P + np.log(Ns) / np.log(xi_0)
    SHIFT = 100 * np.finfo(float).tiny**(1 / P)
    BACKSHIFT = 0.95 * Ns**(1 / Q) * SHIFT**(P / Q)
    # check for bypass option
    if baseplate == 'X':
        return x, sensitivities  # Return as-is
    # determine rotation based on baseplate orientation
    nRot = 'SWNE'.find(baseplate.upper())
    x = np.rot90(x, nRot).copy()
    # initialize xi
    xi = np.zeros_like(x)
    nely, nelx = x.shape
    # loop for applying AM filter from top moving layer-wise downwards
    xi[-1, :] = x[-1,:].copy()
    Xi, keep,sq = [np.zeros_like(x) for i in np.arange(3)]
    for i in np.arange(nely-2,-1,-1):
        cbr = np.pad(xi[i+1,:] + SHIFT,
                     (1, 1),
                     'constant',
                     constant_values=SHIFT)
        keep[i,:] = (cbr[:-2]**P + cbr[1:-1]**P + cbr[2:]**P)
        Xi[i,:] = keep[i,:]**(1 / Q) - BACKSHIFT
        sq[i,:] = np.sqrt((x[i,:] - Xi[i,:])**2 + ep)
        xi[i,:] = 0.5 * ((x[i,:] + Xi[i,:]) - sq[i,:] + np.sqrt(ep))
    # process sensitivities if provided.
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
            # rearrange data for quick multiplication
            qs = dmx.flatten()
            dsmaxdxi = csc_array((qs[1:-1],(qi[1:-1], qj[1:-1])),
                                   shape=(nelx, nelx))
            # update sensitivities
            for k in np.arange(nSens):
                dfx[i,:,k] = dsmindx * (dfxi[i,:,k] + lambda_vals[:,k])
                lambda_vals[:,k] = ((dfxi[i,:,k] + lambda_vals[:,k]) * dsmindXi) @ dsmaxdxi
        # base layer
        dfx[-1,:,:] = dfxi[-1,:,:]+lambda_vals[:,:]
    if sensitivities is None:
        # rotate xi back to original orientation if rotated
        return np.rot90(xi, -nRot)
    else:
        # rotate sensitivities back to original orientation if rotated
        return np.rot90(dfx, -nRot)

def visualise_filter(n,
                     apply_filter,
                     geo=None,
                     fig_kws=None):
    """
    Apply filter to a given geometry and display the original geometry next to
    the filtered one in order to understand the effect of a filter on a given
    geometry of design densities.

    Parameters
    ----------
    n : tuple
        contains number of elements in x,y and z direction depending on number 
        of dimensions.
    apply_filter : callable
        function that applies filter.
    geo : callable or np.ndarray of shape(np.prod(n)) or None
        geometry of design densities on which to apply filter. 
    fig_kws : dict or None, optional
        keywords for figure.

    Returns
    -------
    None.

    """
    #
    ndim = len(n)
    #
    nelx,nely,nelz = n[:ndim] + (None,None,None)[ndim:]
    #
    if geo is None:
        geo = diracdelta(nelx=nelx ,nely=nely, nelz=nelz,
                         location=None )[:,None]
    elif callable(geo):
        geo = diracdelta(nelx=nelx ,nely=nely, nelz=nelz,
                         location=None )
    elif isinstance(geo, np.ndarray):
        geo.shape[0] = int(np.prod(n))
    #
    if ndim == 2:
        # default plot settings 2d
        if fig_kws is None:
            fig_kws = {"figsize": (8,8)}
        #
        fig,axs = subplots(1,2,**fig_kws)
        #
        axs[0].imshow(1-map_eltoimg(quant=geo,
                                    nelx=nelx, nely=nely),
                      cmap="grey")
        #
        filtered = map_eltoimg(quant=apply_filter(geo),
                                    nelx=nelx, nely=nely)
        axs[1].imshow(1-filtered,
                      cmap="grey")
        for i in range(2):
            axs[i].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelbottom=False,
                               labelleft=False)
            axs[i].axis("off")
    elif ndim == 3:
        # default plot settings 3d
        if fig_kws is None:
            fig_kws = {"figsize": figaspect(2.)}
        #
        fig = figure(**fig_kws)
        # unfiltered
        axs = []
        axs.append(fig.add_subplot(2, 1, 1, projection='3d'))
        axs.append(fig.add_subplot(2, 1, 2, projection='3d'))
        #
        dirac_voxel = map_eltovoxel(geo,
                                    nelx=nelx, nely=nely, nelz=nelz)
        #
        facecolors = np.ones(dirac_voxel.shape[:-1] + (4,))
        facecolors[:,:,:,:-1] = 1 - dirac_voxel
        facecolors[:,:,:,-1] = dirac_voxel[:,:,:,0]
        #
        axs[0].voxels(filled = ~np.isclose(dirac_voxel[:,:,:,0], 0),
                      facecolors=facecolors)
        # filtered
        filtered_voxel = map_eltovoxel(quant=apply_filter(geo),
                                       nelx=nelx, nely=nely, nelz=nelz)
        #
        facecolors = np.ones(filtered_voxel.shape[:-1] + (4,))
        facecolors[:,:,:,:-1] = 1 - filtered_voxel
        facecolors[:,:,:,-1] = filtered_voxel[:,:,:,0]
        #
        axs[1].voxels(filled = ~np.isclose(filtered_voxel[:,:,:,0], 0),
                      facecolors=facecolors)
        #
        for i in range(2):
            # limits
            for j,nel in enumerate(n):
                axs[i].set_xlim(0,nel)
            #    
            axs[i].set_xlabel( "z" )
            axs[i].set_ylabel( "y" )
            axs[i].set_zlabel( "x" )
        #
    print("mass before filter operation: ", geo.sum(),"\n",
          "mass after filter operation: ", filtered_voxel.sum())

    show()
    return