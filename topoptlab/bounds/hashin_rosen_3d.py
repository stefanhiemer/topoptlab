# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def heatexp_binary_upp_dx(x : np.ndarray,
                          Kmin: float, Kmax: float,
                          Gmin: float, Gmax: float,
                          amin: float, amax: float) -> np.ndarray:
    """
    Derivative for  the upper Hashin Rosen bound for the heat expansion 
    coefficient of a composite consisting of two isotropic materials.

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
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a_upp_dx : np.ndarray, shape (n)
        derivative of upper bound of heat expansion coefficient.

    """
    return amax - amin + 4*Gmax*(Kmin-Kmax)*(amin-amax) / \
           ( (3*Kmin*Kmax) + (4*Gmax*( (1-x)*Kmin + x*Kmax)) ) * \
           ( 1-2*x - \
            (1-x)*x / ( (3*Kmin*Kmax) + (4*Gmax*( (1-x)*Kmin + x*Kmax)) ) *\
            (4*Gmax*( Kmax - Kmin )) )

def heatexp_binary_upp(x : np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float,
                       amin: float, amax: float) -> np.ndarray:
    """
    Return the upper Hashin Rosen bound for the heat expansion coefficient 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 2.27 of 
    
    "Rosen, B. Walter, and Zvi Hashin. "Effective thermal expansion coefficients 
    and specific heats of composite materials." International Journal of 
    Engineering Science 8.2 (1970): 157-173."

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
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a_upp : np.ndarray, shape (n)
        upper bound of heat expansion coefficient.

    """
    return x*amax + (1-x)*amin + 4*(1-x)*x*Gmax*(Kmin-Kmax)*(amin-amax) / \
           ( (3*Kmin*Kmax) + (4*Gmax*( (1-x)*Kmin + x*Kmax))  )

def heatexp_binary_low_dx(x : np.ndarray,
                          Kmin: float, Kmax: float,
                          Gmin: float, Gmax: float,
                          amin: float, amax: float) -> np.ndarray:
    """
    Derivative for  the lower Hashin Rosen bound for the heat expansion 
    coefficient of a composite consisting of two isotropic materials.

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
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a_low_dx : np.ndarray, shape (n)
        derivative of lower bound of heat expansion coefficient.

    """
    return amax - amin +\
        4*Gmin*(Kmin-Kmax)*(amin-amax) / ( (3*Kmin*Kmax) + (4*Gmin*( (1-x)*Kmin + x*Kmax))  )\
         *(1 - 2*x  - 4*Gmin*(Kmax - Kmin)*x*(1-x) \
           / ( (3*Kmin*Kmax) + (4*Gmin*( (1-x)*Kmin + x*Kmax))  ) )

def heatexp_binary_low(x : np.ndarray,
                       Kmin: float, Kmax: float,
                       Gmin: float, Gmax: float,
                       amin: float, amax: float) -> np.ndarray:
    """
    Return the lower Hashin Rosen bound for the heat expansion coefficient 
    of a composite consisting of two isotropic materials with well ordered 
    moduli (Gmin and Kmin belong to same phase). Taken from Eq. 2.27 of 
    
    "Rosen, B. Walter, and Zvi Hashin. "Effective thermal expansion coefficients 
    and specific heats of composite materials." International Journal of 
    Engineering Science 8.2 (1970): 157-173."

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
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a_low : np.ndarray, shape (n)
        lower bound of heat expansion coefficient.

    """
    return x*amax + (1-x)*amin + 4*x*(1-x)*Gmin*(Kmin-Kmax)*(amin-amax) / \
           ( (3*Kmin*Kmax) + (4*Gmin*( (1-x)*Kmin + x*Kmax))  )
