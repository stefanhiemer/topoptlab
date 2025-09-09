# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def poiss_nary_upp(x: np.ndarray, 
                   Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the Poisson's ratio 
    of a composite consisting of m isotropic materials with well ordered 
    moduli (Gmax and Kmax belong to same phase) as stated in 
    
    Zimmerman, Robert W. "Hashin-Shtrikman bounds on the Poisson ratio of a 
    composite material." Mechanics research communications 19.6 (1992): 563-569.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    nu_upp : np.ndarray, shape (n)
        upper bound of Poisson's ratio

    """
    Ku = bulkmod_nary_upp(x=x, Ks=Ks, Gs=Gs)
    Gl = shearmod_nary_low(x=x, Ks=Ks, Gs=Gs)
    return (3*Ku-2*Gl) / (6*Ku + 2*Gl)

def poiss_nary_upp_dx(x: np.ndarray, 
                      Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the Poisson's 
    ratio of a composite consisting of m isotropic materials with well ordered 
    moduli (Gmax and Kmax belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    nu_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of Poisson's ratio

    """
    Ku = bulkmod_nary_upp(x=x, Ks=Ks, Gs=Gs)[:,None]
    Kudx = bulkmod_nary_upp_dx(x=x, Ks=Ks, Gs=Gs)
    Gl = shearmod_nary_low(x=x, Ks=Ks, Gs=Gs)[:,None]
    Gldx = shearmod_nary_low_dx(x=x, Ks=Ks, Gs=Gs)
    return 1/(6*Ku + 2*Gl) *\
           ( 3*Kudx-2*Gldx - (3*Ku-2*Gl)*(6*Kudx + 2*Gldx)/(6*Ku + 2*Gl) )

def poiss_nary_low(x: np.ndarray, 
                   Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the Poisson's ratio 
    of a composite consisting of m isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase) as stated in 
    
    Zimmerman, Robert W. "Hashin-Shtrikman bounds on the Poisson ratio of a 
    composite material." Mechanics research communications 19.6 (1992): 563-569.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    nu_low : np.ndarray, shape (n)
        lower bound of Poisson's ratio

    """
    Kl = bulkmod_nary_low(x=x, Ks=Ks, Gs=Gs)
    Gu = shearmod_nary_upp(x=x, Ks=Ks, Gs=Gs)
    return (3*Kl-2*Gu) / (6*Kl + 2*Gu)

def poiss_nary_low_dx(x: np.ndarray, 
                      Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the Poisson's 
    ratio of a composite consisting of m isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    nu_low_dx : np.ndarray, shape (n)
        derivative of lower bound of Poisson's ratio

    """
    Kl = bulkmod_nary_low(x=x, Ks=Ks, Gs=Gs)[:,None]
    Kldx = bulkmod_nary_low_dx(x=x, Ks=Ks, Gs=Gs)
    Gu = shearmod_nary_upp(x=x, Ks=Ks, Gs=Gs)[:,None]
    Gudx = shearmod_nary_upp_dx(x=x, Ks=Ks, Gs=Gs)
    return 1/(6*Kl + 2*Gu)*\
           (3*Kldx-2*Gudx - (3*Kl-2*Gu)*(6*Kldx + 2*Gudx) / (6*Kl + 2*Gu) )

def emod_nary_upp(x: np.ndarray, 
                  Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the Young's modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    E_upp : np.ndarray, shape (n)
        upper bound of Young's modulus

    """
    Ku = bulkmod_nary_upp(x=x, Ks=Ks, Gs=Gs)
    Gu = shearmod_nary_upp(x=x, Ks=Ks, Gs=Gs)
    return 9*Ku*Gu / (3*Ku + Gu)

def emod_nary_upp_dx(x: np.ndarray, 
                     Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return derivative of the upper Hashin Shtrikman bound for the Young's 
    modulus of a composite consisting of two isotropic materials with well 
    ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    E_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of Young's modulus

    """
    Ku = bulkmod_nary_upp(x=x, Ks=Ks, Gs=Gs)[:,None]
    Kudx = bulkmod_nary_upp_dx(x=x, Ks=Ks, Gs=Gs)
    Gu = shearmod_nary_upp(x=x, Ks=Ks, Gs=Gs)[:,None]
    Gudx = shearmod_nary_upp_dx(x=x, Ks=Ks, Gs=Gs)
    return 9/(3*Ku+Gu) * (Kudx*Gu + Ku*Gudx - Ku*Gu*(3*Kudx + Gudx)/(3*Ku+Gu) )

def emod_nary_low(x: np.ndarray, 
                  Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the Young's modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    E_low : np.ndarray, shape (n)
        lower bound of Young's modulus

    """
    Kl = bulkmod_nary_low(x=x, Ks=Ks, Gs=Gs)
    Gl = shearmod_nary_low(x=x, Ks=Ks, Gs=Gs)
    return 9*Kl*Gl / (3*Kl + Gl)

def emod_nary_low_dx(x: np.ndarray, 
                     Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the Young's 
    modulus of a composite consisting of two isotropic materials with well 
    ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    E_low_dx : np.ndarray, shape (n)
        derivative of lower bound of Young's modulus

    """
    Kl = bulkmod_nary_low(x=x, Ks=Ks, Gs=Gs)[:,None]
    Kldx = bulkmod_nary_low_dx(x=x, Ks=Ks, Gs=Gs)
    Gl = shearmod_nary_low(x=x, Ks=Ks, Gs=Gs)[:,None]
    Gldx = shearmod_nary_low_dx(x=x, Ks=Ks, Gs=Gs)
    return 9/(3*Kl+Gl) * (Kldx*Gl + Kl*Gldx - Kl*Gl*(3*Kldx + Gldx)/(3*Kl+Gl) )

def poiss_binary_upp(x: np.ndarray,
                     Kmin: float, Kmax: float,
                     Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the Poisson's ratio 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase) as stated in 
    
    Zimmerman, Robert W. "Hashin-Shtrikman bounds on the Poisson ratio of a 
    composite material." Mechanics research communications 19.6 (1992): 563-569.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    nu_upp : np.ndarray, shape (n)
        upper bound of Poisson's ratio

    """
    Ku = bulkmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gl = shearmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    return (3*Ku-2*Gl) / (6*Ku + 2*Gl)

def poiss_binary_upp_dx(x: np.ndarray,
                        Kmin: float, Kmax: float,
                        Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return thederivative of the upper Hashin Shtrikman bound for the Poisson's 
    ratio of a composite consisting of two isotropic materials with well 
    ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    nu_upp : np.ndarray, shape (n)
        upper bound of Poisson's ratio

    """
    Ku = bulkmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Kudx = bulkmod_binary_upp_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                 Gmin=Gmin, Gmax=Gmax)
    Gl = shearmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gldx = shearmod_binary_low_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                  Gmin=Gmin, Gmax=Gmax)
    return 1/(6*Ku + 2*Gl) *\
           ( 3*Kudx-2*Gldx - (3*Ku-2*Gl)*(6*Kudx + 2*Gldx)/(6*Ku + 2*Gl) )

def poiss_binary_low(x: np.ndarray,
                     Kmin: float, Kmax: float,
                     Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the Poisson's ratio 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase) as stated in 
    
    Zimmerman, Robert W. "Hashin-Shtrikman bounds on the Poisson ratio of a 
    composite material." Mechanics research communications 19.6 (1992): 563-569.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    nu_low : np.ndarray, shape (n)
        lower bound of Poisson's ratio

    """
    Kl = bulkmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gu = shearmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    return (3*Kl-2*Gu) / (6*Kl + 2*Gu)

def poiss_binary_low_dx(x: np.ndarray,
                        Kmin: float, Kmax: float,
                        Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the deriviative of the lower Hashin Shtrikman bound for the 
    Poisson's ratio of a composite consisting of two isotropic materials with 
    well ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    nu_low_dx : np.ndarray, shape (n)
        derivative of lower bound of Poisson's ratio

    """
    Kl = bulkmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Kldx = bulkmod_binary_low_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                  Gmin=Gmin, Gmax=Gmax)
    Gu = shearmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gudx = shearmod_binary_upp_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                  Gmin=Gmin, Gmax=Gmax)
    return 1/(6*Kl + 2*Gu)*\
           (3*Kldx-2*Gudx - (3*Kl-2*Gu)*(6*Kldx + 2*Gudx) / (6*Kl + 2*Gu) )

def emod_binary_upp(x: np.ndarray,
                    Kmin: float, Kmax: float,
                    Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the Young's modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    E_upp : np.ndarray, shape (n)
        upper bound of Young's modulus

    """
    Ku = bulkmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gu = shearmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    return 9*Ku*Gu / (3*Ku + Gu)

def emod_binary_upp_dx(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the 
    Young's modulus of a composite consisting of two isotropic materials with 
    well ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    E_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of Young's modulus

    """
    Ku = bulkmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Kudx = bulkmod_binary_upp_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                 Gmin=Gmin, Gmax=Gmax)
    Gu = shearmod_binary_upp(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gudx = shearmod_binary_upp_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                  Gmin=Gmin, Gmax=Gmax)
    return  9/(3*Ku+Gu) * (Kudx*Gu + Ku*Gudx - Ku*Gu*(3*Kudx + Gudx)/(3*Ku+Gu) )

def emod_binary_low(x: np.ndarray,
                    Kmin: float, Kmax: float,
                    Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the Young's modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    E_low : np.ndarray, shape (n)
        lower bound of Young's modulus

    """
    Kl = bulkmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gl = shearmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    return 9*Kl*Gl / (3*Kl + Gl)

def emod_binary_low_dx(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the Young's 
    modulus of a composite consisting of two isotropic materials with well 
    ordered moduli (Gmin and Kmin belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    E_low_dx : np.ndarray, shape (n)
        derivative of lower bound of Young's modulus

    """
    Kl = bulkmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Kldx = bulkmod_binary_low_dx(x=x, Kmin=Kmin, Kmax=Kmax, 
                                  Gmin=Gmin, Gmax=Gmax)
    Gl = shearmod_binary_low(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    Gldx = shearmod_binary_low_dx(x=x, Kmin=Kmin, Kmax=Kmax, Gmin=Gmin, Gmax=Gmax)
    return 9/(3*Kl+Gl) * (Kldx*Gl + Kl*Gldx - Kl*Gl*(3*Kldx + Gldx)/(3*Kl+Gl) )

def shearmod_nary_upp(x: np.ndarray,
                      Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the shear modulus 
    of a composite consisting of m isotropic materials. At the moment I am 
    not aware of any assumption about the ordering of the phases. Taken from 
    Eq. 3.45 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the elastic behaviour of multiphase materials." Journal of the Mechanics 
    and Physics of Solids 11.2 (1963): 127-140."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape (m)
        bulk moduli
    Gs : np.ndarray, shape (m)
        shear moduli

    Returns
    -------
    G_upp : np.ndarray, shape (n)
        upper bound of shear modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmax(Gs)
    Kmax = Ks.max()
    Gmax = Gs[ind]
    # get indices for summation
    G_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    betan = - 3 * (Kmax + 2*Gmax ) / ( 5*Gmax * (3*Kmax + 4*Gmax) )
    # general case
    if mask.any():
        Bn = x[:,x_inds[mask]] / ( 1/ (2*(Gs[None,G_inds[mask]] - Gmax)) - betan )
        Bn = Bn.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        Bn = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        Bn += (1-x.sum(axis=1)) / ( 1/(2*(Gs[-1] - Gmax)) - betan )
    #
    return Gmax + (Bn / (1 + (betan*Bn) ))/2

def shearmod_nary_upp_dx(x: np.ndarray,
                         Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the shear 
    modulus of a composite consisting of m isotropic materials.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    G_upp_dx : np.ndarray, shape (n,m-1)
        derivative of upper bound of shear modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmax(Gs)
    Kmax = Ks.max()
    Gmax = Gs[ind]
    # get indices for summation
    G_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    betan = - 3 * (Kmax + 2*Gmax ) / ( 5*Gmax * (3*Kmax + 4*Gmax) )
    # general case
    if mask.any():
        Bn = x[:,x_inds[mask]] / ( 1/ (2*(Gs[None,G_inds[mask]] - Gmax)) - betan )
        Bn = Bn.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        Bn = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        Bn += (1-x.sum(axis=1)) / ( 1/(2*(Gs[-1] - Gmax)) - betan )
    # Bndx
    # general case
    Bndx = np.zeros(x.shape)
    if mask.any():
        Bndx[:,x_inds[mask]] = 1 / ( 1/(2*(Gs[None,G_inds[mask]] - Gmax)) - betan )
    # case minimum is not the last phase
    if not mask.all():
        Bndx -= 1 / ( 1/(2*(Gs[-1] - Gmax)) - betan )
    #
    return Bndx/(1 + (betan*Bn[:,None]) ) * \
            ( 1 - (Bn[:,None]*betan) / (1 + (betan*Bn[:,None]) ) )/2

def shearmod_nary_low(x: np.ndarray,
                      Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the shear modulus 
    of a composite consisting of m isotropic materials. At the moment I am 
    not aware of any assumption about the ordering of the phases. Taken from 
    Eq. 3.44 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the elastic behaviour of multiphase materials." Journal of the Mechanics 
    and Physics of Solids 11.2 (1963): 127-140."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    G_low : np.ndarray, shape (n)
        lower bound of shear modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmin(Gs)
    Kmin = Ks.min()
    Gmin = Gs[ind]
    # get indices for summation
    G_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    beta1 = - 3 * (Kmin + 2*Gmin ) / ( 5 * Gmin * (3*Kmin + 4*Gmin) )
    # general case
    if mask.any():
        B1 = x[:,x_inds[mask]] / ( 1/ (2*(Gs[None,G_inds[mask]] - Gmin)) - beta1 )
        B1 = B1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        B1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        B1 += (1-x.sum(axis=1)) / ( 1/ (2*(Gs[-1] - Gmin)) - beta1 )
    #
    return Gmin + (B1 / (1 + (beta1*B1) ))/2

def shearmod_nary_low_dx(x: np.ndarray,
                         Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the shear 
    modulus of a composite consisting of m isotropic materials.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    G_low : np.ndarray, shape (n,m-1)
        derivative of lower bound of shear modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmin(Gs)
    Kmin = Ks.min()
    Gmin = Gs[ind]
    # get indices for summation
    G_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    beta1 = - 3 * (Kmin + 2*Gmin ) / ( 5 * Gmin * (3*Kmin + 4*Gmin) )
    # general case
    if mask.any():
        B1 = x[:,x_inds[mask]] / ( 1/ (2*(Gs[None,G_inds[mask]] - Gmin)) - beta1 )
        B1 = B1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        B1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        B1 += (1-x.sum(axis=1)) / ( 1/ (2*(Gs[-1] - Gmin)) - beta1 )
    # B1dx
    # general case
    B1dx = np.zeros(x.shape)
    if mask.any():
        B1dx[:,x_inds[mask]] = 1 / ( 1/(2*(Gs[None,G_inds[mask]] - Gmin)) - beta1 )
    # case minimum is not the last phase
    if not mask.all():
        B1dx -= 1 / ( 1/(2*(Gs[-1] - Gmin)) - beta1 )
    #
    return B1dx/(1 + (beta1*B1[:,None]) ) * \
            ( 1 - (B1[:,None]*beta1) / (1 + (beta1*B1[:,None]) ) )/2

def bulkmod_nary_upp(x: np.ndarray,
                     Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the bulk modulus 
    of a composite consisting of m isotropic materials. At the moment I am 
    not aware of any assumption about the ordering of the phases. Taken from 
    Eq. 3.38 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the elastic behaviour of multiphase materials." Journal of the Mechanics 
    and Physics of Solids 11.2 (1963): 127-140."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    K_upp : np.ndarray, shape (n)
        upper bound of bulk modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find maximum
    ind = np.argmax(Ks)
    Kmax = Ks[ind]
    Gmax = Gs.max()
    # get indices for summation
    K_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alphan = -3 / (3*Kmax + 4*Gmax)
    # general case
    if mask.any():
        An = x[:,x_inds[mask]] / ( 1/(Ks[None,K_inds[mask]] - Kmax) - alphan )
        An = An.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        An = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        An += (1-x.sum(axis=1)) / ( 1/(Ks[-1] - Kmax) - alphan )
    #
    return Kmax + (An / (1 + (alphan*An) ))

def bulkmod_nary_upp_dx(x: np.ndarray,
                        Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the bulk 
    modulus of a composite consisting of m isotropic materials.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        shear moduli

    Returns
    -------
    K_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of bulk modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find maximum
    ind = np.argmax(Ks)
    Kmax = Ks[ind]
    Gmax = Gs.max()
    # get indices for summation
    K_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alphan = -3 / (3*Kmax + 4*Gmax)
    # general case
    if mask.any():
        An = x[:,x_inds[mask]] / ( 1/(Ks[None,K_inds[mask]] - Kmax) - alphan )
        An = An.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        An = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        An += (1-x.sum(axis=1)) / ( 1/(Ks[-1] - Kmax) - alphan )
    # A1dx
    # general case
    Andx = np.zeros(x.shape)
    if mask.any():
        Andx[:,x_inds[mask]] = 1 / ( 1/(Ks[None,K_inds[mask]] - Kmax) - alphan )
    # case minimum is not the last phase
    if not mask.all():
        Andx -= 1 / ( 1/(Ks[-1] - Kmax) - alphan )
    return Andx/(1 + (alphan*An[:,None]) ) * \
            ( 1 - (An[:,None]*alphan) / (1 + (alphan*An[:,None]) ) )

def bulkmod_nary_low(x: np.ndarray,
                     Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the bulk modulus 
    of a composite consisting of m isotropic materials. At the moment I am 
    not aware of any assumption about the ordering of the phases. Taken from 
    Eq. 3.37 of 
    
    "Hashin, Zvi, and Shmuel Shtrikman. "A variational approach to the theory 
    of the elastic behaviour of multiphase materials." Journal of the Mechanics 
    and Physics of Solids 11.2 (1963): 127-140."

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        larger bulk modulus

    Returns
    -------
    K_low : np.ndarray, shape (n)
        lower bound of bulk modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmin(Ks)
    Kmin = Ks[ind]
    Gmin = Gs.min()
    # get indices for summation
    K_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alpha1 = -3 / (3*Kmin + 4*Gmin)
    # general case
    if mask.any():
        A1 = x[:,x_inds[mask]] / ( 1/(Ks[None,K_inds[mask]] - Kmin) - alpha1 )
        A1 = A1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        A1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        A1 += (1-x.sum(axis=1)) / ( 1/(Ks[-1] - Kmin) - alpha1 )
    #
    return Kmin + (A1 / (1 + (alpha1*A1) ))

def bulkmod_nary_low_dx(x: np.ndarray,
                        Ks: np.ndarray, Gs: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the  
    bulk modulus of a composite consisting of m isotropic materials.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    Ks : np.ndarray, shape(m)
        bulk moduli
    Gs : np.ndarray, shape(m)
        larger bulk modulus

    Returns
    -------
    K_low_dx : np.ndarray, shape (n,m-1)
        derivative of lower bound of bulk modulus

    """
    # shapes and indices
    m = Ks.shape[0]
    # find minimum
    ind = np.argmin(Ks)
    Kmin = Ks[ind]
    Gmin = Gs.min()
    # get indices for summation
    K_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    x_inds = np.hstack((np.arange(ind),np.arange(ind+1,m)))
    mask = x_inds < m - 1
    # intermediary stuff
    alpha1 = -3 / (3*Kmin + 4*Gmin)
    # A1
    # general case
    if mask.any():
        A1 = x[:,x_inds[mask]] / ( 1/(Ks[None,K_inds[mask]] - Kmin) - alpha1 )
        A1 = A1.sum(axis=1)
    # case binary and minimum is 2nd phase
    else:
        A1 = np.zeros(x.shape[0])
    # case minimum is not the last phase
    if not mask.all():
        A1 += (1-x.sum(axis=1)) / ( 1/(Ks[-1] - Kmin) - alpha1 )
    # A1dx
    # general case
    A1dx = np.zeros(x.shape)
    if mask.any():
        A1dx[:,x_inds[mask]] = 1 / ( 1/(Ks[None,K_inds[mask]] - Kmin) - alpha1 )
    # case minimum is not the last phase
    if not mask.all():
        A1dx -= 1 / ( 1/(Ks[-1] - Kmin) - alpha1 )
    return A1dx/(1 + (alpha1*A1[:,None]) ) * \
            ( 1 - (A1[:,None]*alpha1) / (1 + (alpha1*A1[:,None]) ) )

def shearmod_binary_upp(x: np.ndarray,
                        Kmin: float, Kmax: float,
                        Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the shear modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 26 of 
    
    "Hashin, Z., and S. Shtrikman. "Hashin, Z., and S. Shtrikman. "Note on a 
    variational approach to the theory of composite elastic materials." Journal 
    of the Franklin Institute 271.4 (1961): 336-341."

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    G_upp : np.ndarray, shape (n)
        upper bound of shear modulus

    """
    return Gmax + (1-x) / ( 1 / (Gmin - Gmax)  + \
            ( 6/5 * ((Kmax +  2*Gmax)*x) / ( (3*Kmax + 4*Gmax)*Gmax) ) )

def shearmod_binary_upp_dx(x: np.ndarray,
                           Kmin: float, Kmax: float,
                           Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the shear 
    modulus of two isotropic materials with well ordered moduli (Gmin and Kmin 
    belong to same phase).

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    G_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of shear modulus

    """
    return -1 / ( 1 / (Gmin - Gmax)  + \
            ( 6/5 * ((Kmax +  2*Gmax)*x) / ( (3*Kmax + 4*Gmax)*Gmax) ) ) -\
           (1-x) / ( 1 / (Gmin - Gmax)  + \
                  ( 6/5 * ((Kmax +  2*Gmax)*x) / ( (3*Kmax + 4*Gmax)*Gmax) ) )**2 * \
            ( 6/5 * ((Kmax +  2*Gmax)) / ( (3*Kmax + 4*Gmax)*Gmax) ) 

def shearmod_binary_low(x: np.ndarray,
                        Kmin: float, Kmax: float,
                        Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the shear modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 27 of 
    
    "Hashin, Z., and S. Shtrikman. "Hashin, Z., and S. Shtrikman. "Note on a 
    variational approach to the theory of composite elastic materials." Journal 
    of the Franklin Institute 271.4 (1961): 336-341."

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    G_low : np.ndarray, shape (n)
        lower bound of shear modulus

    """
    return Gmin + x / ( 1 / (Gmax - Gmin) + \
            ( 6/5 * ((Kmin +  2*Gmin)*(1-x)) / ( (3*Kmin + 4*Gmin)*Gmin) ) )

def shearmod_binary_low_dx(x: np.ndarray,
                           Kmin: float, Kmax: float,
                           Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the shear 
    modulus of two isotropic materials with well ordered moduli (Gmin and Kmin 
    belong to same phase). 

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    G_low_dx : np.ndarray, shape (n)
        derivative of lower bound of shear modulus

    """
    return 1 / ( 1 / (Gmax - Gmin) + \
            ( 6/5 * ((Kmin +  2*Gmin)*(1-x)) / ( (3*Kmin + 4*Gmin)*Gmin) ) ) +\
           x / ( 1 / (Gmax - Gmin) + \
                      ( 6/5 * ((Kmin +  2*Gmin)*(1-x)) / ( (3*Kmin + 4*Gmin)*Gmin) ) )**2 *\
            ( 6/5 * ((Kmin +  2*Gmin)) / ( (3*Kmin + 4*Gmin)*Gmin) ) 

def bulkmod_binary_upp(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the bulk modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 23 of 
    
    "Hashin, Z., and S. Shtrikman. "Hashin, Z., and S. Shtrikman. "Note on a 
    variational approach to the theory of composite elastic materials." Journal 
    of the Franklin Institute 271.4 (1961): 336-341."

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    K_upp : np.ndarray, shape (n)
        upper bound of bulk modulus

    """
    return Kmax + (1-x) / ( 1 / (Kmin - Kmax)  + (3*x / (3 * Kmax + 4*Gmax)) )

def bulkmod_binary_upp_dx(x: np.ndarray,
                          Kmin: float, Kmax: float,
                          Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the bulk 
    modulus of two isotropic materials with well ordered moduli (Gmin and Kmin 
    belong to same phase). 

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    K_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of bulk modulus

    """
    return -1 / ( 1 / (Kmin - Kmax)  + (3*x / (3 * Kmax + 4*Gmax)) ) -\
           (1-x) / ( 1 / (Kmin - Kmax)  + (3*x / (3 * Kmax + 4*Gmax)) )**2 *\
           3 / (3 * Kmax + 4*Gmax)

def bulkmod_binary_low(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the bulk modulus 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 20 of 
    
    "Hashin, Z., and S. Shtrikman. "Hashin, Z., and S. Shtrikman. "Note on a 
    variational approach to the theory of composite elastic materials." Journal 
    of the Franklin Institute 271.4 (1961): 336-341."

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    K_low : np.ndarray, shape (n)
        lower bound of bulk modulus

    """
    return Kmin + x / ( 1 / (Kmax - Kmin)  + (3*(1-x) / (3 * Kmin + 4*Gmin)) )

def bulkmod_binary_low_dx(x: np.ndarray,
                          Kmin: float, Kmax: float,
                          Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the bulk 
    modulus. 

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density of stronger phase
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    Gmin : float
        smaller shear modulus
    Gmax : float
        larger shear modulus

    Returns
    -------
    K_low_dx : np.ndarray, shape (n)
        derivative of lower bound of bulk modulus

    """
    return 1 / ( 1 / (Kmax - Kmin)  + (3*(1-x) / (3 * Kmin + 4*Gmin)) ) +\
           x / ( 1 / (Kmax - Kmin)  + (3*(1-x) / (3 * Kmin + 4*Gmin)) )**2 *\
           3 / (3 * Kmin + 4*Gmin)

def conductivity_nary_upp(x: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the  conductvity 
    of a composite consisting of m isotropic materials. Also applies to 
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
    # Am
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

def conductivity_nary_upp_dx(x: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the  
    conductvity of a composite consisting of m isotropic materials.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    ks : np.ndarray, shape(m)
        conductivities

    Returns
    -------
    k_upp_dx : np.ndarray, shape (n)
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
    # Amdx
    Amdx = np.zeros(x.shape)
    if mask.any():
        Amdx[:,x_inds[mask]] = 1 / ( 1/(ks[None,k_inds[mask]] - kmax) + alpham )
    # case maximum is not the last phase
    if not mask.all():
        Amdx -= 1 / ( 1/(ks[-1] - kmax) + alpham )
    #
    return Amdx / (1 - (alpham*Am[:,None])) * \
           (1 + ((Am[:,None] * alpham)/(1 - (alpham*Am[:,None])) ))

def conductivity_nary_low(x: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the  conductvity 
    of a composite consisting of m isotropic materials. Also applies to 
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

def conductivity_nary_low_dx(x: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the  
    conductvity of a composite consisting of m isotropic materials. 

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
    A1dx = np.zeros(x.shape)
    if mask.any():
        A1dx[:,x_inds[mask]] = 1 / ( 1/(ks[None,k_inds[mask]] \
                                                      - kmin) + alpha1 )
    # case minimum is not the last phase
    if not mask.all():
        A1dx -= 1 / ( 1/(ks[-1] - kmin) + alpha1 )
    #
    return A1dx / (1 - (alpha1*A1[:,None])) * \
           (1 + ((A1[:,None] * alpha1)/(1 - (alpha1*A1[:,None])) ))

def conductivity_binary_upp(x: np.ndarray, 
                            kmin: float, kmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound for the  conductvity 
    of a composite consisting of two isotropic materials. Also applies to 
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

def conductivity_binary_upp_dx(x: np.ndarray, 
                               kmin: float, kmax: float) -> np.ndarray:
    """
    Return the derivative of the upper Hashin Shtrikman bound for the  
    conductvity of a composite consisting of two isotropic materials. 

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

def conductivity_binary_low(x: np.ndarray, 
                            kmin: float, kmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound for the conductvity 
    of a composite consisting of two isotropic materials. Also applies to 
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

def conductivity_binary_low_dx(x: np.ndarray, 
                               kmin: float, kmax: float) -> np.ndarray:
    """
    Return the derivative of the lower Hashin Shtrikman bound for the  
    conductvity of a composite consisting of two isotropic materials. 

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