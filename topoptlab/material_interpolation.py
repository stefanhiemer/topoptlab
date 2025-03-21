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

def heatexpcoeff(kappa, 
                 a1,a2,
                 kappa1,kappa2):
    """
    Return the linear heatexpansion coefficient of a composite consisting of two 
    isotropic substances. 
    
    Rosen & Hasin 1970
    
    Parameters
    ----------
    kappa : np.ndarray, shape (n)
        interpolated bulk modulus
    a1 : np.ndarray, shape (n)
        heat expansion coefficient of material 1
    a2 : np.ndarray, shape (n)
        heat expansion coefficient of material 2
    kappa1 : np.ndarray, shape (n)
        bulk modulus of material 1
    kappa2 : np.ndarray, shape (n)
        bulk modulus of material 2

    Returns
    -------
    a : np.ndarray, shape (n)
        interpolated heat expansion coefficient

    """
    a = (a1 * kappa1 * (kappa2 - kappa) - a2 * kappa2 * (kappa1 - kappa))
    return  a / (kappa * (kappa1-kappa2))
