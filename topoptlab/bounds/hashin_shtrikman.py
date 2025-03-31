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
        relative (volumetric) density of higher conducting phase
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_uppbound : np.ndarray, shape (n)
        upper bound of composite conductivity (or equivalent property)

    """
    # shapes and indices
    m = ks.shape[0]+1
    ind = np.argmax(ks)
    inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    # intermediary stuff
    kmax = ks.max()
    alpham = 1 / (3 * kmax)
    if not (m == 2 and ind == 1):
        Am = x[:, inds] / ( 1/(ks - kmax) + alpham )
        Am = Am.sum(axis=1)
    else:
        A1 = np.zeros(x.shape[0])
    if ind != m - 1:
        Am += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmax) + alpham )
    #
    return kmax + Am / (1 - (alpham*Am) )

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
    x : np.ndarray, shape (n,m)
        relative (volumetric) density of higher conducting phase
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_lowbound : np.ndarray, shape (n)
        lower bound of composite conductivity (or equivalent property)

    """
    # shapes and indices
    m = ks.shape[0]+1
    ind = np.argmin(ks)
    inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    # intermediary stuff
    kmin = ks.min()
    alpha1 = 1 / (3 * kmin)
    if not (m == 2 and ind == 1):
        A1 = (x[:, inds] / ( 1/(ks - kmin) + alpha1 ))
        A1 = A1.sum(axis=1)
    else:
        A1 = np.zeros(x.shape[0])
    if ind != m - 1:
        A1 += (1-x.sum(axis=1)) / ( 1/(ks[-1] - kmin) + alpha1 )
    #
    return kmin + A1 / (1 - (alpha1*A1) )

def conductivity_binary_upp(x,kmin,kmax):
    """
    Return the upper Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 4.8 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the effective magnetic permeability of multiphase materials." Journal of 
    applied Physics 33.10 (1962): 3125-3131."
    

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
    k_uppbound : np.ndarray, shape (n)
        upper bound of composite conductivity (or equivalent property)

    """
    return kmax + (1-x) / ( 1 / (kmin - kmax)  + (x / (3 * kmax)) )

def conductivity_binary_low(x,kmin,kmax):
    """
    Return the lower Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy. Taken from 
    Eq. 4.7 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the effective magnetic permeability of multiphase materials." Journal of 
    applied Physics 33.10 (1962): 3125-3131."

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
    k_lowbound : np.ndarray, shape (n)
        lower bound of composite conductivity (or equivalent property)

    """
    return kmin + x / ( 1 / (kmax - kmin)  + ((1-x) / (3 * kmin)) )
