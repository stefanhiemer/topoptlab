# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np

from topoptlab.utils import to_voigt,safe_inverse,voigt_pair

def eng_density(F: np.ndarray,
                C: Union[None,np.ndarray],
                Fdet: Union[None,np.ndarray],
                h: np.ndarray,
                mu: np.ndarray,
                **kwargs: Any) -> np.ndarray:
    """
    Energy density of Neo-Hookean material:
        
        eng_dens = 1/2 (h*ln(det(F))**2 + mu *((tr(C)-3)-2*ln(det(F))))
        
    C is the right Cauchy–Green deformation tensor
        
        C = F.T @ F
        
    F the deformation gradient, h and mu the first and second Lame constants. 
    det() and inv() are matrix determinant and inverse.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if both C and Fdet are not None.
    C : None or np.ndarray
        Cauchy–Green deformation tensor in matrix notation (...,ndim,ndim).
    Fdet : np.ndarray
        determinant of deformation gradient.
    h : np.ndarray
        first Lame constant.
    mu : np.ndarray
        second Lame constant.
    
    Returns
    -------
    eng_density : np.ndarray
        strain energy density of shape (...).
    """
    #
    if C is None:
        #
        C = F.swapaxes(-1,-2) @ F 
    #
    if Fdet is None:
        #
        Fdet = np.linalg.det(F)
    #
    return (h*np.log(Fdet)**2 + mu *((np.trace(C, axis1=-2, axis2=-1)-3)-2*np.log(Fdet)))/2

def stress_2pk(F: np.ndarray,
               Cinv: Union[None,np.ndarray],
               C: Union[None,np.ndarray],
               Fdet: Union[None,np.ndarray],
               h: np.ndarray,
               mu: np.ndarray,
               **kwargs: Any) -> np.ndarray:
    """
    2. Piola Kirchhoff stress (2PK) in Voigt notation of Neo-Hookean material:
        
        S = h*ln(det(F)) inv(C) + mu *(I-inv(C))
        
    E is the Green-Lagrangian strain tensor in Voigt notation and c the 
    stiffness tensor also in Voigt notation.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if both C and Fdet are not None.
    Cinv : None or np.ndarray
        inverse of Cauchy–Green deformation tensor in matrix notation 
        (...,ndim,ndim).
    C : None or np.ndarray
        Cauchy–Green deformation tensor in matrix notation (...,ndim,ndim).
    Fdet : np.ndarray
        determinant of deformation gradient.
    h : np.ndarray
        first Lame constant.
    mu : np.ndarray
        second Lame constant.
    
    Returns
    -------
    s : np.ndarray
        2. Piola Kirchhoff stress in Voigt notation (..., ndim*(ndim+1)/2).
    """
    #
    if Cinv is None and C is None:
        #
        Cinv = safe_inverse(A=F.swapaxes(-1,-2)@F)
    elif Cinv is None and C is not None:
        #
        Cinv = safe_inverse(A=C)
    #
    if Fdet is None:
        #
        Fdet = np.linalg.det(F)
    #
    I = np.eye(Cinv.shape[-1])\
        .reshape(tuple([1])*len(Cinv.shape[:-2])+Cinv.shape[-2:])
    return to_voigt(h*np.log(Fdet)*Cinv + mu *(I-Cinv))

def consttensor_2pk(F: np.ndarray,
                    Cinv: Union[None,np.ndarray],
                    C: Union[None,np.ndarray],
                    Fdet: Union[None,np.ndarray],
                    h: np.ndarray,
                    mu: np.ndarray,
                    **kwargs: Any) -> np.ndarray:
    """
    2. Piola Kirchhoff stress (2PK) in Voigt notation of Neo-Hookean material:
        
        d S / d E = h*np.kron(inv(C), inv(C)) + \ 
                    2*(mu-h*ln(det(F))) outer(inv(C),inv(C))
        
    E is the Green-Lagrangian strain tensor in Voigt notation and c the 
    stiffness tensor also in Voigt notation:  
        
        outer(A,B)_ijkl​=(A_ik ​B_jl​+ A_il B_jk​)/2
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    Cinv : None or np.ndarray
        inverse of Cauchy–Green deformation tensor in matrix notation 
        (...,ndim,ndim).
    C : None or np.ndarray
        Cauchy–Green deformation tensor in matrix notation (...,ndim,ndim).
    Fdet : np.ndarray
        determinant of deformation gradient.
    h : np.ndarray
        first Lame constant.
    mu : np.ndarray
        second Lame constant.
    
    Returns
    -------
    c : np.ndarray
        constitutive tensor (...,ndim*(ndim+1)/2,ndim*(ndim+1)/2). For this 
        special case, returns a constant.
    """
    #
    if Cinv is None and C is None:
        #
        Cinv = safe_inverse(A=F.swapaxes(-1,-2)@F)
    elif Cinv is None and C is not None:
        #
        Cinv = safe_inverse(A=C)
    #
    if Fdet is None:
        #
        Fdet = np.linalg.det(F) 
    #c_1 = h*np.einsum('...ij,...kl->...ijkl', Cinv, Cinv)
    ##batch_kron(A=Cinv,B=Cinv) + \
    c = h[...,None,None]*np.einsum('...ij,...kl->...ijkl', Cinv, Cinv)+\
           (mu-h*np.log(Fdet))[...,None,None]*\
               (np.einsum('...ik,...jl->...ijkl', Cinv, Cinv) \
               + np.einsum('...il,...jk->...ijkl', Cinv, Cinv))
    
    # convert to Voigt
    ndim = Cinv.shape[-1]
    a, b = np.meshgrid(np.arange(ndim*(ndim+1)//2), np.arange(ndim*(ndim+1)//2), indexing="ij")
    i,j = voigt_pair(a, ndim); k,l = voigt_pair(b, ndim) 
    #
    sf = np.array([1 if i == j else 2
               for i,j in zip(*voigt_pair(np.arange(int(ndim*(ndim+1)/2)), ndim))])
    sf = sf.reshape((1,)*len(c.shape[:-4]) + (sf.size, 1))
    sg = sf.swapaxes(-2, -1)
    return c[..., i, j, k, l] * sg

def neohookean_matmodel(F: np.ndarray,
                        h: np.ndarray,
                        mu: np.ndarray,
                        **kwargs: Any) -> np.ndarray:
    """
    Return 2. Piola Kirchhoff stress (2PK) in Voigt notation and constitutive 
    tensor of Neo-Hookean material.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    h : np.ndarray
        first Lame constant.
    mu : np.ndarray
        second Lame constant.
    
    
    Returns
    -------
    s : np.ndarray
        2. Piola Kirchhoff stress in Voigt notation (..., ndim*(ndim+1)/2).
    c : np.ndarray
        constitutive tensor (...,ndim*(ndim+1)/2,ndim*(ndim+1)/2). For this 
        special case, returns a constant.
    """
    #
    _h = h[..., None, None]
    _mu = mu[..., None, None]
    #
    Cinv = safe_inverse(A=F.swapaxes(-1,-2)@F)
    Fdet = np.linalg.det(F)[...,None,None] 
    # 
    const = consttensor_2pk(F=None,
                            C=None,
                            Cinv=Cinv,
                            Fdet=Fdet,
                            h=_h,mu=_mu)
    s = stress_2pk(F=None,Fdet=Fdet,C=None,Cinv=Cinv,h=_h,mu=_mu)
    return s, const
