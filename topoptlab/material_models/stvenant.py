# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

import numpy as np

from topoptlab.voigt import to_voigt

def eng_density(F: np.ndarray,
                E: Union[None,np.ndarray],
                c: np.ndarray,
                **kwargs: Any) -> np.ndarray:
    """
    Energy density of St. Venant material:
        
        eng_dens = 1/2 E.T @ c @ E
        
    E is the Green-Lagrangian strain tensor in Voigt notation and c the 
    stiffness tensor also in Voigt notation.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    E : None or np.ndarray
        Green-Lagrangian strain tensor in Voigt notation (...,ndim*(ndim+1)/2).
    c : np.ndarray
        stiffness tensor in Voigt notation.
    
    Returns
    -------
    eng_density : np.ndarray
        strain energy density of shape (...).
    """
    #
    if E is None:
        #
        E = to_voigt(F.swapaxes(-1,-2) @ F - np.eye(F.shape[-1])\
                     .reshape( len(F.shape[:-2])*tuple([1]) + F.shape[-2:]), 
                     eng_conv=True)/2
    #
    return 1/2 * E.swapaxes(-1,-2)@c@E

def stress_2pk(F: np.ndarray,
               E: Union[None,np.ndarray],
               c: np.ndarray,
               **kwargs: Any) -> np.ndarray:
    """
    2. Piola Kirchhoff stress (2PK) in Voigt notation of St. Venant material:
        
        S = c @ E
        
    E is the Green-Lagrangian strain tensor in Voigt notation and c the 
    stiffness tensor also in Voigt notation.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    E : None or np.ndarray
        Green-Lagrangian strain tensor in Voigt notation (...,ndim*(ndim+1)/2).
    c : np.ndarray
        stiffness tensor in Voigt notation.
    
    Returns
    -------
    s : np.ndarray
        2. Piola Kirchhoff stress in Voigt notation (..., ndim*(ndim+1)/2).
    """
    #
    if E is None:
        #
        E = to_voigt(F.swapaxes(-1,-2) @ F - np.eye(F.shape[-1])\
                     .reshape( len(F.shape[:-2])*[1] + F.shape[-2:]), 
                     eng_conv=True)/2
    #
    return (c@E[...,None])[...,0]

def consttensor_2pk(F: np.ndarray,
                    c: np.ndarray,
                    **kwargs: Any) -> np.ndarray:
    """
    2. Piola Kirchhoff stress (2PK) in Voigt notation of St. Venant material:
        
        d S_i / d E_j = c_ij 
        
    E is the Green-Lagrangian strain tensor in Voigt notation and c the 
    stiffness tensor also in Voigt notation.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    c : np.ndarray
        stiffness tensor in Voigt notation.
    
    Returns
    -------
    c : np.ndarray
        constitutive tensor (...,ndim*(ndim+1)/2,ndim*(ndim+1)/2). For this 
        special case, returns a constant.
    """
    #
    if len(c.shape) == 2: # assume shape (ndim*(ndim+1)/2,ndim*(ndim+1)/2 
        return np.tile(c,F.shape[:-2]+tuple([1,1]))
    elif len(c.shape) == 3: # assume shape (nel,ndim*(ndim+1)/2,ndim*(ndim+1)/2).
        return np.repeat(c[:,None,:,:],F.shape[1],axis=1)
    else:
        return c

def stvenant_matmodel(F: np.ndarray,
                      c: np.ndarray,
                      **kwargs: Any) -> np.ndarray:
    """
    Return 2. Piola Kirchhoff stress (2PK) in Voigt notation and constitutive 
    tensor of St. Venant material.
    
    Parameters
    ----------
    F : np.ndarray
        deformation gradient in matrix form of shape (...,ndim,ndim). Ignored 
        if E is not None.
    c : np.ndarray
        stiffness tensor in Voigt notation.
    
    Returns
    -------
    s : np.ndarray
        2. Piola Kirchhoff stress in Voigt notation (..., ndim*(ndim+1)/2).
    c : np.ndarray
        constitutive tensor (...,ndim*(ndim+1)/2,ndim*(ndim+1)/2). For this 
        special case, returns a constant.
    """
    #
    E = to_voigt(F.swapaxes(-1,-2) @ F - np.eye(F.shape[-1])\
                 .reshape( len(F.shape[:-2])*tuple([1]) + F.shape[-2:]), 
                 eng_conv=True)/2
    const = consttensor_2pk(F=F,c=c)
    return stress_2pk(F=F,E=E,c=const), const
