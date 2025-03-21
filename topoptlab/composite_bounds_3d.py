def conductivity_hashin_shtrikman_upp(x, 
                                      kmin,kmax):
    """
    Return the upper Hashin Shtrikman bound for the thermal conductvity 
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
    return kmax + (1-x) / ( 1 / (kmin - kmax)  + (x / (3 * kmax)) )

def conductivity_hashin_shtrikman_low(x, 
                                      kmin,kmax):
    """
    Return the lower Hashin Shtrikman bound for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy.

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
        lower bound of composite shear modulus

    """
    return kmin + x / ( 1 / (kmax - kmin)  + ((1-x) / (3 * kmin)) )
