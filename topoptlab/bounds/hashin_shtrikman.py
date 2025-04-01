import numpy as np

def conductivity_nary_upp(x,ks):
    """
    Return the upper Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of m isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 3.22 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the effective magnetic permeability of multiphase materials." Journal of 
    applied Physics 33.10 (1962): 3125-3131."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_upp : np.ndarray, shape (n)
        upper bound of composite conductivity (or equivalent property)

    """
    # shapes and indices
    m = ks.shape[0]
    ind = np.argmax(ks)
    # find maximum
    ind = np.argmax(ks)
    kmax = ks[ind]
    # get indices for summation
    k_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alpham = 1 / (3 * kmax)
    # general case
    if mask.any():
        Am = x[:,x_inds[mask]] / ( 1/(ks[None,k_inds[mask]] - kmax) + alpham )
        Am = Am.sum(axis=1)
    # case binary and maximum is 2nd phase
    else:
        Am = np.zeros(x.shape[0])
    # case maximum is not the last phase
    if not mask.all():
        Am += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmax) + alpham )
    #
    return kmax + Am / (1 - (alpham*Am) )

def conductivity_nary_low_dx(x,ks):
    """
    Return the derivative of the lower Hashin Shtrikman bound for the thermal 
    conductvity of a composite consisting of two isotropic substances. 

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_low_dx : np.ndarray, shape (n)
        lower bound of composite conductivity (or equivalent property)

    """
    # shapes and indices
    m = ks.shape[0]
    # find minimum
    ind = np.argmin(ks)
    kmin = ks[ind]
    # get indices for summation
    k_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alpha1 = 1 / (3 * kmin)
    # A1
    # general case
    if mask.any():
        A1 = x[:,x_inds[mask]] / ( 1/(ks[None,k_inds[mask]] - kmin) + alpha1 )
        A1 = A1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        A1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        A1 += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmin) + alpha1 )
    # dA1/dx
    if mask.any():
        A1dx = x[:,x_inds[mask]] / ( 1/(ks[None,k_inds[mask]] - kmin) + alpha1 )
    # case binary and minimum is 2nd phase
    else:
        A1dx = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        A1dx += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmin) + alpha1 )
    #
    return kmin + (A1 / (1 - (alpha1*A1) ))

def conductivity_nary_low(x,ks):
    """
    Return the lower Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of m isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 3.21 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the effective magnetic permeability of multiphase materials." Journal of 
    applied Physics 33.10 (1962): 3125-3131."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_low : np.ndarray, shape (n)
        lower bound of composite conductivity (or equivalent property)

    """
    # shapes and indices
    m = ks.shape[0]
    # find minimum
    ind = np.argmin(ks)
    kmin = ks[ind]
    # get indices for summation
    k_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alpha1 = 1 / (3 * kmin)
    # general case
    if mask.any():
        A1 = x[:,x_inds[mask]] / ( 1/(ks[None,k_inds[mask]] - kmin) + alpha1 )
        A1 = A1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        A1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        A1 += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmin) + alpha1 )
    #
    return kmin + (A1 / (1 - (alpha1*A1) ))

def conductivity_binary_upp_dx(x,kmin,kmax):
    """
    Return the derivative of the upper Hashin Shtrikman bound for the thermal 
    conductvity of a composite consisting of two isotropic substances. 

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    kmin : float
        smaller conductivity
    kmax : float
        larger conductivity

    Returns
    -------
    k_upp_dx : np.ndarray, shape (n)
        derivative upper bound of composite conductivity (or equivalent 
        property).

    """
    return -1 / ( 1 / (kmin - kmax)  + (x / (3 * kmax)) ) - \
           (1-x) / ( 1 / (kmin - kmax)  + (x / (3 * kmax)) )**2 * 1/(3*kmax) 

def conductivity_binary_upp(x,kmin,kmax):
    """
    Return the upper Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 16 of 
    
    "Hashin, Z., and S. Shtrikman. "Note on the effective constants of 
    composite materials." Journal of the Franklin institute 271.5 (1961): 
    423-426."
    

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    kmin : float
        smaller conductivity
    kmax : float
        larger conductivity

    Returns
    -------
    k_upp : np.ndarray, shape (n)
        upper bound of composite conductivity (or equivalent property)

    """
    return kmax + (1-x) / ( 1 / (kmin - kmax)  + (x / (3 * kmax)) )

def conductivity_binary_low_dx(x,kmin,kmax):
    """
    Return the derivative of the lower Hashin Shtrikman bound for the thermal 
    conductvity of a composite consisting of two isotropic substances. 

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of higher conducting phase
    kmin : float
        smaller conductivity
    kmax : float
        larger conductivity

    Returns
    -------
    k_low_dx : np.ndarray, shape (n)
        derivative of lower bound of composite conductivity (or equivalent 
        property)

    """
    return 1 / ( 1 / (kmax - kmin)  + ((1-x) / (3 * kmin)) ) + \
           x / ( 1 / (kmax - kmin)  + ((1-x) / (3 * kmin)) )**2 * 1/(3*kmin) 

def conductivity_binary_low(x,kmin,kmax):
    """
    Return the lower Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 13 (in the original script the label is written as 31) of 
    
    "Hashin, Z., and S. Shtrikman. "Note on the effective constants of 
    composite materials." Journal of the Franklin institute 271.5 (1961): 
    423-426."

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of higher conducting phase
    kmin : float
        smaller conductivity
    kmax : float
        larger conductivity

    Returns
    -------
    k_low : np.ndarray, shape (n)
        lower bound of composite conductivity (or equivalent property)

    """
    return kmin + x / ( 1 / (kmax - kmin)  + ((1-x) / (3 * kmin)) )
