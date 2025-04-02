def bulkmodulus_hashin_shtrikman_upp(x, 
                                     bulkm_min,bulkm_max,
                                     shearm_min,shearm_max):
    """
    Return the upper Hashin Shtrikman bound in 2D for the bulkmodulus of 
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
    bulkm_uppbound : np.ndarray, shape (n)
        upper bound of composite bulk modulus

    """
    return (1-x) * bulkm_min + x * bulkm_max - \
           (1-x) * x * (bulkm_max-bulkm_min)**2\
           /((1-x)*bulkm_max + x*bulkm_min + shearm_max)
           
def bulkmodulus_hashin_shtrikman_low(x, 
                                     bulkm_min,bulkm_max,
                                     shearm_min,shearm_max):
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
    return (1-x) * bulkm_min + x * bulkm_max - \
           (1-x) * x * (bulkm_max-bulkm_min)**2\
           /((1-x)*bulkm_max + x*bulkm_min + shearm_min)

def shearmodulus_hashin_shtrikman_upp(x, 
                                     bulkm_min,bulkm_max,
                                     shearm_min,shearm_max):
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
    return (1-x) * shearm_min + x * shearm_max - \
           (1-x) * x * (shearm_max-shearm_min)**2\
           /((1-x)*shearm_max + x*shearm_min + \
             shearm_max * bulkm_max / (shearm_max + 2*bulkm_max))
               
def shearmodulus_hashin_shtrikman_low(x, 
                                     bulkm_min,bulkm_max,
                                     shearm_min,shearm_max):
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
    return (1-x)*shearm_min + x * shearm_max - \
           (1-x)*x* (shearm_max - shearm_min)**2\
           /((1-x)*shearm_max + x*shearm_min + \
             shearm_min*bulkm_min / (shearm_min + 2*bulkm_min)) 

def _conductivity_hashin_shtrikman_upp(x, 
                                      k_min,k_max):
    """
    Return the upper Hashin Shtrikman bound in 2D for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    k_min : float
        smaller conductivity
    k_max : float
        larger conductivity

    Returns
    -------
    k_uppbound : np.ndarray, shape (n)
        upper bound of composite shear modulus

    """
    return (1-x)*k_min + x*k_max - \
           (1-x) * x * (k_max-k_min)**2\
           /((1-x)*k_max + x*k_min + k_max) 

def _conductivity_hashin_shtrikman_low(x, 
                                      k_min,k_max):
    """
    Return the lower Hashin Shtrikman bound in 2D for the thermal conductvity 
    of a composite consisting of two isotropic substances. Also applies to 
    electrical conductivity and diffusion by concept of analogy.

    Parameters
    ----------
    x : np.ndarray, shape (n)
        relative density
    k_min : float
        smaller conductivity
    k_max : float
        larger conductivity

    Returns
    -------
    k_lowbound : np.ndarray, shape (n)
        lower bound of composite shear modulus

    """
    return (1-x)*k_min + x*k_max - \
           (1-x) * x * (k_max-k_min)**2\
           /((1-x)*k_max + x*k_min + k_min)
