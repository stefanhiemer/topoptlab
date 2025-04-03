def simp(xPhys, eps, penal):
    """
    Return scale factor for the modified SIMP method (Sigmund 2007)..

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    eps : float
        volume fraction.
    penal : float
        penalty exponent for the SIMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property

    """
    return eps+(1-eps)*(xPhys)**penal

def simp_dp(xPhys, eps, penal):
    """
    Return the derivative of the scale factor for the modified SIMP method 
    (Sigmund 2007) with regards to the SIMP density.

    Parameters
    ----------
    xPhys : np.ndarray, shape (n)
        physical SIMP density
    eps : float
        volume fraction.
    penal : float
        penalty exponent for the SIMP method.

    Returns
    -------
    scale : np.ndarray, shape (n)
        scale factor to multiply with material property

    """
    return penal * (1-eps)*(xPhys)**(penal-1)

def heatexpcoeff_binary_iso(x, K, 
                            a1, a2,
                            K1, K2):
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
    x : np.ndarray, shape (n)
        volume fraction of phase 1
    K : np.ndarray, shape (n)
        interpolated / effective bulk modulus
    a1 : np.ndarray, shape (n)
        heat expansion coefficient of material 1
    a2 : np.ndarray, shape (n)
        heat expansion coefficient of material 2
    K1 : np.ndarray, shape (n)
        bulk modulus of material 1
    K2 : np.ndarray, shape (n)
        bulk modulus of material 2

    Returns
    -------
    a : np.ndarray, shape (n)
        interpolated heat expansion coefficient

    """
    return x*a1 + (1-x)*a2 + (a1-a2)*K1*K2/(K2 - K1) * (1/K - x/K1 + (1-x)/K2)
