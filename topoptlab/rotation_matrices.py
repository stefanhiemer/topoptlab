# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

def R_2d(theta: np.ndarray) -> np.ndarray:
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

def dR2_dtheta(theta: np.ndarray) -> np.ndarray:
    """
    Derivative of R_2d w.r.t. theta

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    dR : np.ndarray, shape (n,2,2)
        derivatives of rotation matrices.
    """
    return np.column_stack((
        -np.sin(theta), -np.cos(theta),
         np.cos(theta), -np.sin(theta)
    )).reshape((theta.shape[0], 2, 2))
 
def Rv_2d(theta: np.ndarray, eng_conv: bool = False) -> np.ndarray:
    """
    2D rotation matrix in Voigt notation.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radians
    eng_conv : bool
        False -> standard Voigt [xx, yy, xy]
        True  -> engineering Voigt [xx, yy, gamma_xy]

    Returns
    -------
    Rv : np.ndarray, shape (n,3,3)
    """
    if not eng_conv:
        return np.column_stack((np.cos(theta)**2, np.sin(theta)**2, -2.0 * np.cos(theta) * np.sin(theta),
                                np.sin(theta)**2, np.cos(theta)**2, 2.0 * np.cos(theta) * np.sin(theta),
                                np.cos(theta) * np.sin(theta), -np.cos(theta) * np.sin(theta), np.cos(theta)**2 - np.sin(theta)**2
                                )).reshape((theta.shape[0], 3, 3))
    else:
        return np.column_stack((np.cos(theta)**2, np.sin(theta)**2, -np.cos(theta) * np.sin(theta),
                                np.sin(theta)**2, np.cos(theta)**2, np.cos(theta) * np.sin(theta),
                                2.0 * np.cos(theta) * np.sin(theta), -2.0 * np.cos(theta) * np.sin(theta),np.cos(theta)**2 - np.sin(theta)**2
                                )).reshape((theta.shape[0], 3, 3))


def dRvdtheta_2d(theta: np.ndarray, eng_conv: bool = False) -> np.ndarray:
    """
    Derivative of the 2D Voigt rotation matrix with respect to theta.

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radians
    eng_conv : bool
        False -> standard Voigt [xx, yy, xy]
        True  -> engineering Voigt [xx, yy, gamma_xy]

    Returns
    -------
    dRv : np.ndarray, shape (n,3,3)
    """
    theta = np.asarray(theta).reshape(-1)

    if not eng_conv:
        return np.column_stack((-2.0 * np.cos(theta) * np.sin(theta), 2.0 * np.cos(theta) * np.sin(theta), -2.0 * (np.cos(theta)**2 - np.sin(theta)**2),
                                2.0 * np.cos(theta) * np.sin(theta), -2.0 * np.cos(theta) * np.sin(theta), 2.0 * (np.cos(theta)**2 - np.sin(theta)**2),
                                (np.cos(theta)**2 - np.sin(theta)**2), -(np.cos(theta)**2 - np.sin(theta)**2), -4.0 * np.cos(theta) * np.sin(theta)
                                )).reshape((theta.shape[0], 3, 3))
    else:
        return np.column_stack((-2.0 * np.cos(theta) * np.sin(theta), 2.0 * np.cos(theta) * np.sin(theta), -(np.cos(theta)**2 - np.sin(theta)**2),
                                2.0 * np.cos(theta) * np.sin(theta), -2.0 * np.cos(theta) * np.sin(theta), (np.cos(theta)**2 - np.sin(theta)**2),
                                2.0 * (np.cos(theta)**2 - np.sin(theta)**2), -2.0 * (np.cos(theta)**2 - np.sin(theta)**2), -4.0 * np.cos(theta) * np.sin(theta)
                                )).reshape((theta.shape[0], 3, 3))

def R_3d(theta: np.ndarray, phi: np.ndarray)-> np.ndarray:
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

def dR3_dtheta(theta: np.ndarray, phi: np.ndarray)-> np.ndarray:
    """
    Derivative of R_3d w.r.t. theta
    """
    return np.column_stack((
        -np.sin(theta)*np.cos(phi), -np.cos(theta), -np.sin(theta)*np.sin(phi),
         np.cos(theta)*np.cos(phi), -np.sin(theta),  np.cos(theta)*np.sin(phi),
         np.zeros(theta.shape[0]),  np.zeros(theta.shape[0]),  np.zeros(theta.shape[0])
    )).reshape((theta.shape[0], 3, 3))

def dR3_dphi(theta: np.ndarray, phi: np.ndarray)-> np.ndarray:
    """
    Derivative of R_3d w.r.t. phi
    """
    return np.column_stack((
        -np.cos(theta)*np.sin(phi),  np.zeros(theta.shape[0]),  np.cos(theta)*np.cos(phi),
        -np.sin(theta)*np.sin(phi),  np.zeros(theta.shape[0]),  np.sin(theta)*np.cos(phi),
        -np.cos(phi),                np.zeros(theta.shape[0]), -np.sin(phi)
    )).reshape((theta.shape[0], 3, 3))


def Rv_3d(theta: np.ndarray, phi: np.ndarray, eng_conv: bool = False) -> np.ndarray:
    """
    3D rotation matrix in Voigt notation.

    Voigt order:
        [xx, yy, zz, yz, xz, xy]

    Parameters
    ----------
    theta : np.ndarray, shape (n,)
        angle in radians
    phi : np.ndarray, shape (n,)
        angle in radians
    eng_conv : bool
        False -> standard Voigt [xx, yy, zz, yz, xz, xy]
        True  -> engineering Voigt [xx, yy, zz, gamma_yz, gamma_xz, gamma_xy]

    Returns
    -------
    Rv : np.ndarray, shape (n,6,6)
    """
    theta = np.asarray(theta).reshape(-1)
    phi = np.asarray(phi).reshape(-1)

    Rv_eng = np.column_stack((
        np.cos(phi)**2*np.cos(theta)**2,
        np.sin(theta)**2,
        np.sin(phi)**2*np.cos(theta)**2,
        -np.sin(phi)*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        -np.sin(theta)*np.cos(phi)*np.cos(theta),

        np.sin(theta)**2*np.cos(phi)**2,
        np.cos(theta)**2,
        np.sin(phi)**2*np.sin(theta)**2,
        np.sin(phi)*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.sin(theta)*np.cos(phi)*np.cos(theta),

        np.sin(phi)**2,
        np.zeros(theta.shape[0]),
        np.cos(phi)**2,
        np.zeros(theta.shape[0]),
        -np.sin(2.0*phi)/2.0,
        np.zeros(theta.shape[0]),

        -np.cos(2.0*phi - theta)/2.0 + np.cos(2.0*phi + theta)/2.0,
        np.zeros(theta.shape[0]),
        np.cos(2.0*phi - theta)/2.0 - np.cos(2.0*phi + theta)/2.0,
        np.cos(phi)*np.cos(theta),
        np.sin(theta)*np.cos(2.0*phi),
        -np.sin(phi)*np.cos(theta),

        -np.sin(2.0*phi - theta)/2.0 - np.sin(2.0*phi + theta)/2.0,
        np.zeros(theta.shape[0]),
        np.sin(2.0*phi - theta)/2.0 + np.sin(2.0*phi + theta)/2.0,
        -np.sin(theta)*np.cos(phi),
        np.cos(2.0*phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),

        2.0*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        -np.sin(2.0*theta),
        2.0*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(2.0*theta),
        np.cos(2.0*phi - 2.0*theta)/4.0 - np.cos(2.0*phi + 2.0*theta)/4.0,
        np.cos(phi)*np.cos(2.0*theta)
    )).reshape((theta.shape[0], 6, 6))

    if eng_conv:
        return Rv_eng

    S = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    S_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    return np.einsum('ij,ejk,kl->eil', S_inv, Rv_eng, S, optimize=True)


def dRvdtheta_3d(theta: np.ndarray, phi: np.ndarray, eng_conv: bool = False) -> np.ndarray:
    """
    Derivative of Rv_3d with respect to theta.
    """
    theta = np.asarray(theta).reshape(-1)
    phi = np.asarray(phi).reshape(-1)

    dRv_eng = np.column_stack((
        -2.0*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        np.sin(2.0*theta),
        -2.0*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        -np.sin(phi)*np.cos(2.0*theta),
        -np.cos(2.0*phi - 2.0*theta)/4.0 + np.cos(2.0*phi + 2.0*theta)/4.0,
        -np.cos(phi)*np.cos(2.0*theta),

        2.0*np.sin(theta)*np.cos(phi)**2*np.cos(theta),
        -np.sin(2.0*theta),
        2.0*np.sin(phi)**2*np.sin(theta)*np.cos(theta),
        np.sin(phi)*np.cos(2.0*theta),
        np.cos(2.0*phi - 2.0*theta)/4.0 - np.cos(2.0*phi + 2.0*theta)/4.0,
        np.cos(phi)*np.cos(2.0*theta),

        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),
        np.zeros(theta.shape[0]),

        -np.sin(2.0*phi - theta)/2.0 - np.sin(2.0*phi + theta)/2.0,
        np.zeros(theta.shape[0]),
        np.sin(2.0*phi - theta)/2.0 + np.sin(2.0*phi + theta)/2.0,
        -np.sin(theta)*np.cos(phi),
        np.cos(theta)*np.cos(2.0*phi),
        np.sin(phi)*np.sin(theta),

        np.cos(2.0*phi - theta)/2.0 - np.cos(2.0*phi + theta)/2.0,
        np.zeros(theta.shape[0]),
        -np.cos(2.0*phi - theta)/2.0 + np.cos(2.0*phi + theta)/2.0,
        -np.cos(phi)*np.cos(theta),
        -np.sin(theta)*np.cos(2.0*phi),
        np.sin(phi)*np.cos(theta),

        2.0*np.cos(phi)**2*np.cos(2.0*theta),
        -2.0*np.cos(2.0*theta),
        2.0*np.sin(phi)**2*np.cos(2.0*theta),
        -2.0*np.sin(phi)*np.sin(2.0*theta),
        np.sin(2.0*phi - 2.0*theta)/2.0 + np.sin(2.0*phi + 2.0*theta)/2.0,
        -2.0*np.sin(2.0*theta)*np.cos(phi)
    )).reshape((theta.shape[0], 6, 6))

    if eng_conv:
        return dRv_eng

    S = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    S_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    return np.einsum('ij,ejk,kl->eil', S_inv, dRv_eng, S, optimize=True)


def dRvdphi_3d(theta: np.ndarray, phi: np.ndarray, eng_conv: bool = False) -> np.ndarray:
    """
    Derivative of Rv_3d with respect to phi.
    """
    theta = np.asarray(theta).reshape(-1)
    phi = np.asarray(phi).reshape(-1)

    dRv_eng = np.column_stack((
        -2.0*np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        np.zeros(theta.shape[0]),
        2.0*np.sin(phi)*np.cos(phi)*np.cos(theta)**2,
        np.sin(phi - 2.0*theta)/4.0 - np.sin(phi + 2.0*theta)/4.0,
        np.cos(2.0*phi)*np.cos(theta)**2,
        np.sin(phi)*np.sin(theta)*np.cos(theta),

        -2.0*np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.zeros(theta.shape[0]),
        2.0*np.sin(phi)*np.sin(theta)**2*np.cos(phi),
        np.sin(theta)*np.cos(phi)*np.cos(theta),
        np.sin(theta)**2*np.cos(2.0*phi),
        -np.cos(phi - 2.0*theta)/4.0 + np.cos(phi + 2.0*theta)/4.0,

        np.sin(2.0*phi),
        np.zeros(theta.shape[0]),
        -np.sin(2.0*phi),
        np.zeros(theta.shape[0]),
        -np.cos(2.0*phi),
        np.zeros(theta.shape[0]),

        2.0*(2.0*np.sin(phi)**2 - 1.0)*np.sin(theta),
        np.zeros(theta.shape[0]),
        -np.sin(2.0*phi - theta) + np.sin(2.0*phi + theta),
        -np.sin(phi)*np.cos(theta),
        -2.0*np.sin(2.0*phi)*np.sin(theta),
        -np.cos(phi)*np.cos(theta),

        2.0*(2.0*np.sin(phi)**2 - 1.0)*np.cos(theta),
        np.zeros(theta.shape[0]),
        np.cos(2.0*phi - theta) + np.cos(2.0*phi + theta),
        np.sin(phi)*np.sin(theta),
        -2.0*np.sin(2.0*phi)*np.cos(theta),
        np.sin(theta)*np.cos(phi),

        -np.cos(2.0*phi - 2.0*theta)/2.0 + np.cos(2.0*phi + 2.0*theta)/2.0,
        np.zeros(theta.shape[0]),
        np.cos(2.0*phi - 2.0*theta)/2.0 - np.cos(2.0*phi + 2.0*theta)/2.0,
        np.cos(phi)*np.cos(2.0*theta),
        -np.sin(2.0*phi - 2.0*theta)/2.0 + np.sin(2.0*phi + 2.0*theta)/2.0,
        -np.sin(phi)*np.cos(2.0*theta)
    )).reshape((theta.shape[0], 6, 6))

    if eng_conv:
        return dRv_eng

    S = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    S_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    return np.einsum('ij,ejk,kl->eil', S_inv, dRv_eng, S, optimize=True)