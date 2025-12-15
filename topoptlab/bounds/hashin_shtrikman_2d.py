# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def bulkmod_binary_upp(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound in 2D for the bulkmodulus of 
    a composite consisting of two isotropic substances.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
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
    bulkm_uppbound : np.ndarray, shape (n)
        upper bound of composite bulk modulus

    """
    return (1-x) * Kmin + x * Kmax - \
           (1-x) * x * (Kmax-Kmin)**2\
           /((1-x)*Kmax + x*Kmin + Gmax)
           
def bulkmod_binary_low(x: np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound in 2D for the bulkmodulus of 
    a composite consisting of two isotropic substances.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    bulkm_min : float
        smaller bulk modulus
    bulkm_max : float
        larger bulk modulus
    shearm_min : float
        smaller shear modulus
    shearm_max : float
        larger shear modulus

    Returns
    -------
    bulkm_lowbound : np.ndarray, shape (n)
        lower bound of composite bulk modulus

    """
    return (1-x) * Kmin + x * Kmax - \
           (1-x) * x * (Kmax-Kmin)**2\
           /((1-x)*Kmax + x*Kmin + Gmin)

def shearmod_binary_upp(x: np.ndarray,
                                      Kmin: float, Kmax: float,
                                      Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound in 2D for the shearmodulus of 
    a composite consisting of two isotropic substances.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    bulkm_min : float
        smaller bulk modulus
    bulkm_max : float
        larger bulk modulus
    shearm_min : float
        smaller shear modulus
    shearm_max : float
        larger shear modulus

    Returns
    -------
    shearm_uppbound : np.ndarray, shape (n)
        upper bound of composite shear modulus

    """
    return (1-x) * Gmin + x * Gmax - (1-x) * x * (Gmax-Gmin)**2\
           /((1-x)*Gmax + x*Gmin + Gmax * Kmax / (Gmax + 2*Kmax))
               
def shearmod_binary_low(x: np.ndarray,
                        Kmin: float, Kmax: float,
                        Gmin: float, Gmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound in 2D for the shearmodulus of 
    a composite consisting of two isotropic substances.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    bulkm_min : float
        smaller bulk modulus
    bulkm_max : float
        larger bulk modulus
    shearm_min : float
        smaller shear modulus
    shearm_max : float
        larger shear modulus

    Returns
    -------
    shearm_lowbound : np.ndarray, shape (n)
        lower bound of composite shear modulus

    """
    return (1-x)*Gmin + x * Gmax - (1-x)*x* (Gmax - Gmin)**2\
           /((1-x)*Gmax + x*Gmin + Gmin*Kmin / (Gmin + 2*Kmin)) 

def conductivity_binary_upp(x: np.ndarray, 
                            kmin: float, kmax: float) -> np.ndarray:
    """
    Return the upper Hashin Shtrikman bound in 2D for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy.

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
        upper bound of composite shear modulus

    """
    return (1-x)*kmin + x*kmax - \
           (1-x) * x * (kmax-kmin)**2\
           /((1-x)*kmax + x*kmin + kmax) 

def conductivity_binary_low(x: np.ndarray, 
                            kmin: float, kmax: float) -> np.ndarray:
    """
    Return the lower Hashin Shtrikman bound in 2D for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy.

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
    k_lowbound : np.ndarray, shape (n)
        lower bound of composite shear modulus

    """
    return (1-x)*kmin + x*kmax - (1-x) * x * (kmax-kmin)**2\
           /((1-x)*kmax + x*kmin + kmin)
