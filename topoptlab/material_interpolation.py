# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Dict

import numpy as np

def simp(xPhys: np.ndarray, eps: np.ndarray, penal: float,
         **kwargs: Any) -> np.ndarray:
    """
    Return scale factor for the modified SIMP method by

    Sigmund, Ole. "Morphology-based black and white filters for topology
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007):
    401-424.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    eps : float
        small offset that is needed for some properties (e. g. stiffness) to
        avoid degeneracy of the problem. Should be very small compared to 1
        e. g. 1e-9.
    penal : float
        penalty exponent for the SIMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property/matrix.

    """
    return eps+(1-eps)*xPhys**penal

def simp_dx(xPhys: np.ndarray, eps: np.ndarray, penal: float,
            **kwargs: Any) -> np.ndarray:
    """
    Return the derivative of the scale factor for the modified SIMP method

    Sigmund, Ole. "Morphology-based black and white filters for topology
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007):
    401-424.

    with regards to the SIMP density.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    eps : float
        small offset that is needed for some properties (e. g. stiffness) to
        avoid degeneracy of the problem. Should be very small compared to 1
        e. g. 1e-9.
    penal : float
        penalty exponent for the SIMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property/matrix.

    """
    return penal * (1-eps) * xPhys**(penal-1)

def ramp(xPhys: np.ndarray, eps: np.ndarray, penal: float,
         **kwargs: Any) -> np.ndarray:
    """
    Return scale factor for the RAMP method:

    Stolpe, Mathias, and Krister Svanberg. "An alternative interpolation scheme
    for minimum compliance topology optimization." Structural and
    Multidisciplinary Optimization 22.2 (2001): 116-124.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical RAMP density
    eps : float
        small offset that is needed for some properties (e. g. stiffness) to
        avoid degeneracy of the problem. Should be very small compared to 1
        e. g. 1e-9.
    penal : float
        penalty factor for the RAMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property/matrix.

    """
    return eps+(1-eps)*xPhys/(1+penal*(1-xPhys))

def ramp_dx(xPhys: np.ndarray, eps: np.ndarray, penal: float,
            **kwargs: Any) -> np.ndarray:
    """
    Return the derivative of the scale factor for the RAMP method:

    Stolpe, Mathias, and Krister Svanberg. "An alternative interpolation scheme
    for minimum compliance topology optimization." Structural and
    Multidisciplinary Optimization 22.2 (2001): 116-124.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical RAMP density
    eps : float
        small offset that is needed for some properties (e. g. stiffness) to
        avoid degeneracy of the problem. Should be very small compared to 1
        e. g. 1e-9.
    penal : float
        penalty factor for the RAMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property/matrix.

    """
    return (1+penal)*(1+penal*(1-xPhys))**(-2)

def bound_interpol(xPhys: np.ndarray, w: float,
                   bd_low: Callable, bd_upp: Callable,
                   bd_kws: Dict,
                   **kwargs: Any) -> np.ndarray:
    """
    Interpolate a material property A between a lower and upper bound according
    to

    A_interpol = (1-w) A_low + w A_upp

    This scheme was to my knowledge first suggested in

    Bendsøe, Martin P., and Ole Sigmund. "Material interpolation schemes in
    topology optimization." Archive of applied mechanics 69 (1999): 635-654.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical, relative density of stronger phase
    w : float
        weight to trade-off between lower and upper bound. Must be between 0/1.
    bd_low : callable
        lower bound on property A that takes x and bd_kws as input
        arguments.
    bd_upp : callable
        upper bound on property A that takes xP and bd_kws as input
        arguments.
    bd_kws : dict
        dictionary that contains the keywords necessary for the bound
        functions.

    Returns
    -------
    A : np.ndarray, shape (n)
        scale factor to multiply with material property

    """
    return (1-w)*bd_low(x=xPhys,**bd_kws) + w*bd_upp(x=xPhys,**bd_kws)

def bound_interpol_dx(xPhys: np.ndarray, w: float,
                      bd_low_dx: Callable, bd_upp_dx: Callable,
                      bd_kws: Dict,
                      **kwargs: Any) -> np.ndarray:
    """
    Derivative of bound interpolation with regards to phys. densities

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    w : float
        weight to trade-off between lower and upper bound. Must be between 0/1.
    bd_low_dx : callable
        derivative with regards to xPhys of  lower bound on property A that 
        takes xPhys and bd_kws as input arguments.
    bd_upp_dx : callable
        derivative with regards to xPhys of upper bound on property A that 
        takes xPhys and bd_kws as input arguments.
    bd_kws : dict
        dictionary that contains the keywords necessary for the bound
        functions.

    Returns
    -------
    A : np.ndarray, shape (n)
        scale factor to multiply with material property

    """
    return (1-w)*bd_low_dx(x=xPhys,**bd_kws) + w*bd_upp_dx(x=xPhys,**bd_kws)

def heatexpcoeff_binary_iso(xPhys: np.ndarray, K: np.ndarray,
                            Kmin: float, Kmax: float,
                            amin: float, amax: float) -> np.ndarray:
    """
    Return the linear heatexpansion coefficient of a composite consisting of two
    isotropic substances. Taken from Eq. 2.26 of

    Rosen, B. Walter, and Zvi Hashin. "Effective thermal expansion coefficients
    and specific heats of composite materials." International Journal of
    Engineering Science 8.2 (1970): 157-173.

    As in this equation it is not stated how K* (effective bulk modulus) is
    calculated, I assume it to be provided by some material interpolation
    function (e. g. SIMP). Mind you that this interpolation must lie between
    the Hashin-Shtrikman bounds.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical, relative density of stronger phase
    K : np.ndarray, shape (n)
        interpolated / effective bulk modulus
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a : np.ndarray, shape (n)
        interpolated heat expansion coefficient

    """
    return xPhys*amax + (1-xPhys)*amin + (amin-amax) / (1/Kmin - 1/Kmax) * \
           (1/K - (1-xPhys)/Kmin - xPhys/Kmax )

def heatexpcoeff_binary_iso_dx(xPhys: np.ndarray, 
                               K: np.ndarray, dKdx: np.ndarray,
                               Kmin: float, Kmax: float,
                               amin: float, amax:float) -> np.ndarray:
    """
    Return the derivative of the linear heatexpansion coefficient of a
    composite consisting of two isotropic substances.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical, relative density of stronger phase
    K : np.ndarray, shape (n)
        interpolated / effective bulk modulus
    dKdx : np.ndarray, shape (n)
        derivative of interpolated / effective bulk modulus
    Kmin : float
        smaller bulk modulus
    Kmax : float
        larger bulk modulus
    amin : float
        heat expansion coefficient of weaker phase.
    amax : float
        heat expansion coefficient of stronger phase.

    Returns
    -------
    a : np.ndarray, shape (n)
        interpolated heat expansion coefficient

    """
    return amax - amin + (amin-amax) / (1/Kmin - 1/Kmax) *\
           (-dKdx/K**2 + 1/Kmin - 1/Kmax )
