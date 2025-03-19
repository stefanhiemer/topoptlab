from numpy import array

def isotropic_2d(E=1.,nu=0.3,plane_stress=True):
    """
    2D stiffness tensor for isotropic material. 
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    plane_stress : bool
        if True, return stiffness tensor for plane stress, otherwise return
        stiffness tensor for plane strain
    
    Returns
    -------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    """
    if plane_stress:
        return E/(1-nu**2)*array([[1,nu,0],
                                  [nu,1,0],
                                  [0,0,(1-nu)/2]])
    else:
        return E/((1+nu)*(1-2*nu))*array([[1-nu,nu,0],
                                          [nu,1-nu,0],
                                          [0,0,(1-nu)/2]])

def orthotropic_2d(Ex, Ey, nu_xy, G_xy):
    """
    2D stiffness tensor for isotropic material. 
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson' ratio.
    plane_stress : bool
        if True, return stiffness tensor for plane stress, otherwise return
        stiffness tensor for plane strain
    
    Returns
    -------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    """
    nu_yx = nu_xy * (Ey / Ex)  # Reciprocity relation
    factor = Ex / (Ex - (Ey * nu_xy**2) )
    
    return factor * array([[Ex, nu_xy * Ey, 0],
                           [nu_xy * Ey, Ey, 0],
                           [0, 0, G_xy * (1 - nu_xy * nu_yx)]])

def isotropic_3d(E=1.,nu=0.3):
    """
    3D stiffness tensor for isotropic material. 
    
    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    
    Returns
    -------
    c : np.ndarray, shape (6,6)
        stiffness tensor.
    """
    return E/((1+nu)*(1-2*nu))*array([[1-nu,nu,nu,0,0,0],
                                      [nu,1-nu,nu,0,0,0],
                                      [nu,nu,1-nu,0,0,0],
                                      [0,0,0,(1-nu)/2,0,0],
                                      [0,0,0,0,(1-nu)/2,0],
                                      [0,0,0,0,0,(1-nu)/2]])

def octet_trusslattice(E,rho):
    """
    Octet truss structure from "Effective properties of the octet-truss lattice 
    material".
    
    Parameters
    ----------
    E : float
        Young's modulus.
    rho : float
        relative density.
    
    Returns
    -------
    c : np.ndarray, shape (6,6)
        stiffness tensor.
    """
    return E*rho/3 * array([[1/2,1/4,1/4,0,0,0],
                            [1/4,1/4,1/2,0,0,0],
                            [1/4,1/4,1/2,0,0,0],
                            [0,0,0,1/4,0,0],
                            [0,0,0,0,1/4,0],
                            [0,0,0,0,0,1/4]])

def rank2_2d(mu1,mu2,nu,E):
    """
    Stiffness tensor for rank2 laminate in 2D taken from 
    "Wu, Jun, Ole Sigmund, and Jeroen P. Groen. "Topology optimization of multi-scale structures: a review." Structural and Multidisciplinary Optimization 63 (2021): 1455-1480.".
    
    Parameters
    ----------
    mu1 : float
        layer width first (bigger) layer.
    mu2 : float
        layer width second (smaller) layer.
    nu : float
        Poisson's ratio.
    E : float
        Young's modulus.
    
    Returns
    -------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    """
    factor = E/(1-mu2+mu1*mu2*(1-nu))
    return factor * array([[mu1,mu1*mu2*nu,0],
                           [mu1*mu2*nu,mu2*(1-mu2+mu1*mu2),0],
                           [0,0,0]])