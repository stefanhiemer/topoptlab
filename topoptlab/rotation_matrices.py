# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def R_2d(theta):
    """
    2D rotation matrix

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,2,2)
        rotation matrices.

    """
    return np.column_stack((np.cos(theta),-np.sin(theta),
                            np.sin(theta),np.cos(theta)))\
          .reshape((theta.shape[0],2,2))
          
def Rv_2d(theta):
    """
    2D rotation matrix for tensors of 2nd order ("Voigt vectors") and 4th order 
    ("Voigt matrices"). 

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,3,3)
        rotation matrices.

    """
    
    return np.column_stack((np.cos(theta)**2, np.sin(theta)**2, -np.sin(2*theta)/2, 
                            np.sin(theta)**2, np.cos(theta)**2, np.sin(2*theta)/2, 
                            np.sin(2*theta), -np.sin(2*theta), np.cos(2*theta)))\
          .reshape((theta.shape[0],3,3))

def R_3d(theta, phi):
    """
    3D rotation matrix.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radian for rotation around z axis.
    phi : np.ndarray, shape (n,)
        angle in radian for rotation around y axis.

    Returns
    -------
    R : np.ndarray, shape (n, 3, 3)
        Rotation matrices for each (theta, phi) pair.
    """
    return np.column_stack((np.cos(theta)*np.cos(phi),-np.sin(theta),np.cos(theta)*np.sin(phi),
                            np.sin(theta)*np.cos(phi),np.cos(theta),np.sin(theta)*np.sin(phi),
                            -np.sin(phi),np.zeros(theta.shape[0]),np.cos(phi)))\
          .reshape((theta.shape[0],3,3))

def Rv_3d(theta, phi):
    return np.column_stack((np.cos(phi)**2*np.cos(theta)**2,
                            np.sin(theta)**2,
                            np.sin(phi)**2*np.cos(theta)**2,
                            0,
                            np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
                            0,
                            np.sin(theta)**2*np.cos(phi)**2,
                            np.cos(theta)**2,
                            np.sin(phi)**2*np.sin(theta)**2,
                            0,
                            np.sin(phi)*np.sin(theta)**2*np.cos(phi),
                            0,
                            np.sin(phi)**2,
                            0,
                            np.cos(phi)**2,
                            0,
                            -np.sin(2*phi)/2,
                            0,
                            -np.cos(2*phi - theta)/2 + np.cos(2*phi + theta)/2,
                            0,
                            np.cos(2*phi - theta)/2 - np.cos(2*phi + theta)/2,
                            0,
                            -np.sin(2*phi - theta)/2 + np.sin(2*phi + theta)/2,
                            0,
                            -np.sin(2*phi - theta)/2 - np.sin(2*phi + theta)/2,
                            0,
                            np.sin(2*phi - theta)/2 + np.sin(2*phi + theta)/2,
                            0,
                            np.cos(2*phi - theta)/2 + np.cos(2*phi + theta)/2,
                            0,
                            2*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
                            -np.sin(2*theta),
                            2*np.sin(phi)**2*np.sin(theta)*np.cos(theta), 
                            0, 
                            np.cos(2*phi - 2*theta)/4 - np.cos(2*phi + 2*theta)/4, 
                            0)).reshape((theta.shape[0],6,6))

def dRvdtheta_2d(theta):
    """
    First order derivtive of 2D rotation matrix for tensors of 2nd order 
    ("Voigt vectors") and 4th order ("Voigt matrices"). 

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    dRvdtheta : np.ndarray shape (n,3,3)
        rotation matrices.

    """
    return np.column_stack((-np.sin(2*theta), np.sin(2*theta), -np.cos(2*theta),
                            np.sin(2*theta), -np.sin(2*theta), np.cos(2*theta), 
                            2*np.cos(2*theta), -2*np.cos(2*theta), -2*np.sin(2*theta)))\
          .reshape((theta.shape[0],3,3))