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

def heatexpcoeff(x, 
                 a1,a2,
                 bulkm1,bulkm2):
    """
    Return the linear heatexpansion oefficient of a composite consisting of two 
    isotropic substances. 
    
    Rosen & Hasin 1970
    
    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    a_min : float
        smaller conductivity
    a_max : float
        larger conductivity

    Returns
    -------
    k_uppbound : np.ndarray, shape (n)
        upper bound of composite shear modulus

    """
    raise NotImplementedError("Not yet finished.")
