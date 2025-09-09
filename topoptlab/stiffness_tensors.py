# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple,Union

import numpy as np

def compute_elastic_properties_3d(E: Union[None,float] = None, 
                                  nu: Union[None,float] = None,  
                                  G: Union[None,float] = None,  
                                  K: Union[None,float] = None,  
                                  lam: Union[None,float] = None,
                                  M: Union[None,float] = None
                                  ) -> Tuple[float,float,float,
                                             float,float,float]:
    """
    Compute all 3D isotropic elastic properties from any 2 given elast. 
    properties. 
    
    Parameters
    ----------
    E : None or float
        Young's modulus.
    nu : None or float
        Poisson's ratio.
    G : None or float
        shear modulus.
    K : None or float
        bulk modulus.
    lam : None or float
        Lamé's first parameter.
    M : None or float
        P-wave modulus.
    
    Returns
    -------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    G : float
        shear modulus.
    K : float
        bulk modulus.
    lam : float
        Lamé's first parameter.
    M : float
        P-wave modulus.
    """
    # count how many values are provided
    given = {k: v for k, v in locals().items() if v is not None and k != 'math'}
    if len(given) < 2:
        raise ValueError("Provide at least two independent elastic constants.")
    # calculate missing values based on known pairs
    if K is not None and E is not None:
        nu = (3*K - E) / (6*K)
        G = 3*(K*E) / (9*K - E)
        lam = 3*K*(3*K-E) / (9*K - E)
        M = 3*K*(3*K+E) / (9*K-E)
        return E,nu,G,K,lam,M
    elif K is not None and lam is not None:
        E = 9*K*(K-lam) / (3*K-lam)
        G = 3 * (K-lam) / 2
        nu = lam / (3*K-lam)
        M = 3*K - 2*lam
        return E,nu,G,K,lam,M
    elif K is not None and G is not None:
        E = 9*K*G / (3*K + G)
        nu = (3*K - 2*G) / (2*(3*K + G))
        lam = K - 2/3 * G
        M = K + 4/3 * G
        return E,nu,G,K,lam,M
    elif K is not None and nu is not None:
        E = 3*K*(1-2*nu)
        lam = 3*K*nu / (1+nu)
        G = 3*K*(1-2*nu) / (2*(1+nu))
        M = 3*K*(1-nu) / (1+nu)
        return E,nu,G,K,lam,M
    elif K is not None and M is not None:
        E = 9*K*(M-K) / (3*K+M)
        lam = (3*K - M) /2
        G = 3*(M-K) / 4
        nu = (3*K - M) / (3*K+M)
        return E,nu,G,K,lam,M
    elif E is not None and lam is not None:
        R = np.sqrt(E**2 + 9*lam**2+2*E*lam)
        K = (E + 3*lam + R) / 6
        G = (E - 3*lam + R) / 4
        nu = 2 * lam / (E+lam+R)
        M = (E-lam+R)/2
        return E,nu,G,K,lam,M
    elif E is not None and G is not None:
        nu = E / (2 * G) - 1
        K = E*G / (3 * (3*G-E))
        lam = G*(E-2*G) / (3*G - E)
        M = G*(4*G-E) / (3*G-E)
        return E,nu,G,K,lam,M
    elif E is not None and nu is not None:
        G = E / (2 * (1 + nu))
        K = E / (3 * (1 - 2 * nu))
        lam = E*nu / ( (1+nu)*(1-2*nu) )
        M = E*(1-nu) / ( (1+nu)*(1-2*nu) )
        return E,nu,G,K,lam,M
    elif E is not None and M is not None:
        S = np.sqrt(E**2 + 9*M**2 - (10*E*M))
        G = (3*M+E-S) / 8
        K = (3*M - E + S) / 6
        lam = (M - E + S) / 4
        nu = (E-M+S) / (4*M)
        return E,nu,G,K,lam,M
    elif lam is not None and G is not None:
        E = G*(3*lam + 2*G)/(lam + G)
        nu = lam / (2*(lam + G))
        K = lam + 2/3 * G
        M = lam + 2*G
        return E,nu,G,K,lam,M
    elif lam is not None and nu is not None:
        E =  lam*(1+nu)*(1-2*nu) / nu
        G = lam*(1-2*nu) / (2*nu)
        K = lam*(1+nu) / (3*nu)
        M = lam*(1-nu) / nu
        return E,nu,G,K,lam,M
    elif lam is not None and M is not None:
        K = (M+2*lam) / 3
        E = (M-lam)*(M+2*lam) / (M+lam)
        G = (M-lam) / 2
        nu = lam / (M+lam)
        return E,nu,G,K,lam,M
    elif G is not None and nu is not None:
        K = 2*G*(1+nu) / ( 3*(1-2*nu) )  
        E = 2*G*(1+nu)
        lam = 2*G*nu / (1-2*nu)
        M = 2*G*(1-nu) / (1-2*nu)
        return E,nu,G,K,lam,M
    elif G is not None and M is not None:
        K = M - 4/3 * G
        lam = M-2*G
        E = (G*(3*M-4*G) / (M-G))
        nu = (M-2*G) / (2*M-2*G)
        return E,nu,G,K,lam,M
    elif nu is not None and M is not None:
        K = M*(1+nu) / ( 3*(1-nu) )
        E = M*(1+nu)*(1-2*nu) / (1-nu)
        G = M*(1-2*nu) / (2*(1-nu))
        lam = M*nu / (1-nu)
        return E,nu,G,K,lam,M
    else:
        raise ValueError("Unsupported or insufficient input combination.")

def isotropic_2d(E: float = 1., 
                 nu: float = 0.3, 
                 plane_stress: bool = True) -> np.ndarray:
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
        return E/(1-nu**2)*np.array([[1,nu,0],
                                     [nu,1,0],
                                     [0,0,(1-nu)/2]])
    else:
        return E/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,0],
                                             [nu,1-nu,0],
                                             [0,0,(1-nu)/2]])

def orthotropic_2d(Ex: float, Ey: float, 
                   nu_xy: float, G_xy: float,
                   Ez: Union[None,float] = None, 
                   nu_xz: Union[None,float] = None, 
                   nu_yz: Union[None,float] = None,
                   plane_stress: bool = True) -> np.ndarray:
    """
    2D stiffness tensor for orthotropic material. The indices of the Poisson 
    ratios nu_ij are defined as the direction with the applied strain (i) and 
    the direction of contraction/expansion j. 
    
    Parameters
    ----------
    Ex : float
        Young's modulus in x direction.
    Ey : float
        Young's modulus in y direction.
    G_xy : float
        shear modulus in xy plane.
    nu_xy : float
        Poisson's ratio for tension in x and contraction/expansion y direction.
    Ez : float or None
        Young's modulus in z direction. Only needed for plain strain.
    nu_xz : float or None
        Poisson's ratio for tension in x and contraction/expansion z direction.
        Only needed for plain strain.
    nu_yz : float or None
        Poisson's ratio for tension in y and contraction/expansion z direction.
        Only needed for plain strain.
    
    Returns
    -------
    c : np.ndarray, shape (3,3)
        stiffness tensor.
    """
    #
    nu_yx = nu_xy * (Ey / Ex)
    #
    if plane_stress:
        return np.array([[Ex / (1-nu_xy*nu_yx), Ex*nu_yx / (1-nu_xy*nu_yx) ,0], 
                         [Ey*nu_xy / (1-nu_xy*nu_yx), Ey / (1-nu_xy*nu_yx), 0],
                         [0, 0, G_xy]])
    else:
        nu_zy = nu_yz * (Ez / Ey)
        nu_zx = nu_xz * (Ez / Ex)
        D = 1 - nu_xy*nu_yx - nu_xz*nu_zx - nu_yz*nu_zy - 2*nu_xy*nu_yz*nu_zx
        return np.array([[Ex*(1-nu_yz*nu_zy)/D,Ex*(nu_yx + nu_yz*nu_zx)/D,0], 
                         [Ey*(nu_xy + nu_xz*nu_zy)/D, Ey*(1 - nu_xz*nu_zx)/D, 0],
                         [0, 0, G_xy]])

def orthotropic_3d(Ex: float, Ey: float, Ez: float, 
                   nu_xy: float, nu_xz: float, nu_yz: float,
                   G_xy: float, G_xz: float, G_yz: float) -> np.ndarray:
    """
    3D stiffness tensor for orthotropic material. The indices of the Poisson 
    ratios nu_ij are defined as the direction with the applied strain (i) and 
    the direction of contraction/expansion j. 
    
    Parameters
    ----------
    Ex : float
        Young's modulus in x direction.
    Ey : float
        Young's modulus in y direction.
    Ez : float
        Young's modulus in z direction.
    nu_xy : float
        Poisson's ratio for tension in x and contraction/expansion y direction.
    nu_xz : float
        Poisson's ratio for tension in x and contraction/expansion z direction.
    nu_yz : float
        Poisson's ratio for tension in y and contraction/expansion z direction.
    G_xy : float
        shear modulus in xy plane.
    G_xz : float
        shear modulus in xz plane.
    G_yz : float
        shear modulus in yz plane.
    
    Returns
    -------
    c : np.ndarray, shape (6,6)
        stiffness tensor.
    """
    #
    nu_zy = nu_yz * (Ez / Ey)
    nu_zx = nu_xz * (Ez / Ex)
    nu_yx = nu_xy * (Ey / Ex)
    #
    D = 1 - nu_xy*nu_yx - nu_xz*nu_zx - nu_yz*nu_zy - 2*nu_xy*nu_yz*nu_zx
    return np.array([[Ex*(1-nu_yz*nu_zy)/D,
                      Ex*(nu_yx + nu_yz*nu_zx)/D, 
                      Ex*(nu_zx + nu_zy*nu_yx)/D, 
                      0, 0, 0], 
                     [Ey*(nu_xy + nu_xz*nu_zy)/D, 
                      Ey*(1 - nu_xz*nu_zx)/D,
                      Ey*(nu_zy + nu_zx*nu_xy)/D, 
                      0, 0, 0], 
                     [Ez*(nu_xz + nu_xy*nu_yz)/D, 
                      Ez*(nu_yz + nu_yx*nu_xz)/D, 
                      Ez*(1-nu_xy*nu_yx)/D, 
                      0, 0, 0], 
                     [0, 0, 0, G_yz, 0, 0], 
                     [0, 0, 0, 0, G_xz, 0], 
                     [0, 0, 0, 0, 0, G_xy]])

def isotropic_3d(E:float = 1., nu:float = 0.3) -> np.ndarray:
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
    return E/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,nu,0,0,0],
                                         [nu,1-nu,nu,0,0,0],
                                         [nu,nu,1-nu,0,0,0],
                                         [0,0,0,(1-nu)/2,0,0],
                                         [0,0,0,0,(1-nu)/2,0],
                                         [0,0,0,0,0,(1-nu)/2]])

def octet_trusslattice(E:float, rho:float) -> np.ndarray:
    """
    Octet truss structure from 
    
    "Deshpande, Vikram S., Norman A. Fleck, and Michael F. Ashby. "Effective 
    properties of the octet-truss lattice material." Journal of the Mechanics 
    and Physics of Solids 49.8 (2001): 1747-1769.".
    
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
    return E*rho/3 * np.array([[1/2,1/4,1/4,0,0,0],
                               [1/4,1/4,1/2,0,0,0],
                               [1/4,1/4,1/2,0,0,0],
                               [0,0,0,1/4,0,0],
                               [0,0,0,0,1/4,0],
                               [0,0,0,0,0,1/4]])

def rank2_2d(mu1: float, mu2: float, nu: float, E: float) -> np.ndarray:
    """
    Stiffness tensor for rank2 laminate in 2D taken from 
    
    "Wu, Jun, Ole Sigmund, and Jeroen P. Groen. "Topology optimization of 
    multi-scale structures: a review." Structural and Multidisciplinary 
    Optimization 63 (2021): 1455-1480.".
    
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
    return factor * np.array([[mu1,mu1*mu2*nu,0],
                              [mu1*mu2*nu,mu2*(1-mu2+mu1*mu2),0],
                              [0,0,0]])