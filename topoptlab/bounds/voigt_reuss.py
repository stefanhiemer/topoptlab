# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable

import numpy as np

from topoptlab.utils import svd_inverse,cholesky_inverse

def voigt(x: np.ndarray, 
          props: np.ndarray) -> np.ndarray:
    """
    Return the Voigt bound for a general property which might be either a 
    scalar quantity or in matrix form. The Voigt bound is the
    volume average over all phases for a scalar property:
        
        prop_V = x[:,0]*prop[0]+x[:,1]*prop[1]+...+x[:,m-2]*prop[m-2] + \
                 x.sum(axis=1)*prop[m-1]
                 
    The formula applies the same for matrix properties.

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    props : np.ndarray, shape(m) or shape (m,k,k)
        scalar properties or tensors 

    Returns
    -------
    prop_upp : np.ndarray, shape (n) or shape (n,k,k)
        Voigt bound

    """
    if len(props.shape) == 1:
        return (x*props[None,:-1]).sum(axis=1) + (1-x.sum(axis=1))*props[-1]
    elif len(props.shape) == 3:
        return (x[:,:,None,None]*props[None,:-1,:,:]).sum(axis=1) +\
                (1-x.sum(axis=1))[:,None,None]*props[-1,:,:]
    else:
        raise ValueError("shape of props inconsistent.")

def voigt_dx(x: np.ndarray, 
             props: np.ndarray) -> np.ndarray:
    """
    Return the derivative of the Voigt bound for a general property which might 
    be either a scalar quantity or in matrix form. Each entry is
        
        prop_V_dx[i] = [prop[0]-prop[-1],prop[1]-prop[-1],...,prop[m-2]-prop[-1]]

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    props : np.ndarray, shape(m) or shape (m,k,k)
        scalar properties or tensors.

    Returns
    -------
    prop_upp_dx : np.ndarray, shape (n,m-1) or shape (n,m-1,k,k)
        derivative of Voigt bound.

    """
    if len(props.shape) == 1:
        return np.ones(x.shape)*(props[:-1]-props[-1])[None,:]
    elif len(props.shape) == 3:
        return np.ones(x.shape)[:,:,None,None]*\
               (props[:-1]-props[-1])[None,:,:,:]
    else:
        raise ValueError("shape of props inconsistent.")
    

def reuss(x: np.ndarray, 
          props: np.ndarray,
          inverse: Callable = svd_inverse) -> np.ndarray:
    """
    Return the Reuss bound for a general property which might be either a 
    scalar quantity or in matrix form. The Reuss bound is the harmonic volume 
    average over all phases:
        
        prop_R = x[:,0]/prop[0]+x[:,1]/prop[1]+...+x[:,m-2]/prop[m-2]+\
                 x.sum(axis=1)/prop[m-1]
    
    The formula applies slightly changed for matrix properties via the matrix
    inverse inv():
        
        prop_R = inv(x[:,0]*inv(prop[0])+x[:,1]*inv(prop[1])+...+\
                     x[:,m-2]*inv(prop[m-2])+x.sum(axis=1)*inv(prop[m-1]))

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    props : np.ndarray, shape(m) or shape (m,k,k)
        scalar properties or tensors.
    function to calculate inverse property. Only necessary for matrix 
    properties.

    Returns
    -------
    prop_low: np.ndarray, shape (n) or shape (n,k,k)
        Reuss bound

    """
    if len(props.shape) == 1:
        return ((x/props[None,:-1]).sum(axis=1) + (1-x.sum(axis=1))/props[-1])**(-1)
    elif len(props.shape) == 3:
        #return np.linalg.inv((x[:,:,None,None]*np.linalg.inv(props[:-1])[None,:]).sum(axis=1)+\
        #                     (1-x.sum(axis=1))[:,None,None]*np.linalg.inv(props[-1]))
        return inverse((x[:,:,None,None]*inverse(props[:-1])[None,:]).sum(axis=1)+\
                       (1-x.sum(axis=1))[:,None,None]*inverse(props[-1]))
    else:
        raise ValueError("shape of props inconsistent.")
        
def reuss_dx(x: np.ndarray, 
             props: np.ndarray,
             inverse: Callable = svd_inverse) -> np.ndarray:
    """
    Return the derivative of the Reuss bound for a general property which might 
    be either a scalar quantity or in matrix form. Assume that the Reuss bound
    prop_R has already been calculated as prop_R. Then the derivative becomes
    
        prop_R_dx[i] = (-1) * prop_R[i]@( inv(prop[0]) - inv(prop[-1]) )@prop_R[i]

    Parameters
    ----------
    x : np.ndarray, shape (n,m-1)
        volume fraction of first m-1 phases. The volume fraction of the mth 
        phase can then be inferred via 1-x.sum(axis=1)
    props : np.ndarray, shape(m) or shape (m,k,k)
        scalar properties or tensors 
    inverse : callable
        function to calculate inverse property. Only necessary for matrix 
        properties.

    Returns
    -------
    prop_low_dx: np.ndarray, shape (n,m-1) or shape (n,m-1,k,k)
        derivative of Reuss bound.

    """
    if len(props.shape) == 1:
        return (-1)*((x/props[None,:-1]).sum(axis=1)+(1-x.sum(axis=1))/props[-1])[:,None]**(-2)*\
               (1/props[:-1] - 1/props[-1])[None,:]
    elif len(props.shape) == 3:
        Ainv = inverse((x[:,:,None,None]*inverse(props[:-1])[None,:]).sum(axis=1)+\
                       (1-x.sum(axis=1))[:,None,None]*inverse(props[-1])) 
        return (-1)*np.einsum('nij,mjk,nkl->nmil', 
                              Ainv,
                              inverse(props[:-1])-inverse(props[-1:]),#np.linalg.inv(props[:-1])-np.linalg.inv(props[-1:]),
                              Ainv)
    else:
        raise ValueError("shape of props inconsistent.")