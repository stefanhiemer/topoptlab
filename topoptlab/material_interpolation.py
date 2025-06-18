def simp(xPhys, eps, penal):
    """
    Return scale factor for the modified SIMP method (Sigmund 2007)..

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

def simp_dx(xPhys, eps, penal):
    """
    Return the derivative of the scale factor for the modified SIMP method 
    (Sigmund 2007) with regards to the SIMP density.

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

def bound_interpol(xPhys,w,
                   bd_low,bd_upp,
                   bd_kws):
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

def bound_interpol_dx(xPhys,w,
                      bd_low_dx,bd_upp_dx,
                      bd_kws):
    """
    Derivative of bound interpolation with regards to phys. densities

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    w : float
        weight to trade-off between lower and upper bound. Must be between 0/1.
    penal : float
        penalty exponent for the SIMP method.

    Returns
    -------
    A : np.ndarray, shape (n)
        scale factor to multiply with material property

    """
    return (1-w)*bd_low_dx(x=xPhys,**bd_kws) + w*bd_upp_dx(x=xPhys,**bd_kws)

def heatexpcoeff_binary_iso(xPhys, K,
                            Kmin, Kmax,
                            amin, amax):
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
    return xPhys*amax + (1-xPhys)*amin + (amin-amax) / (1/Kmin - 1/Kmax) * (1/K - (1-xPhys)/Kmin - xPhys/Kmax )
           
def heatexpcoeff_binary_iso_dx(xPhys, K, dKdx,
                               Kmin, Kmax,
                               amin, amax):
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