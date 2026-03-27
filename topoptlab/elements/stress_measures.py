# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Tuple
import numpy as np

def von_mises_stress(stress: np.ndarray, ndim: int) -> np.ndarray:
    """
    Compute element-wise von Mises stress.

    Parameters
    ----------
    stress : np.ndarray, shape (nels, 3) in 2D or (nels, 6) in 3D
        Stress in Voigt notation.
        2D ordering: [sxx, syy, sxy]
        3D ordering: [sxx, syy, szz, syz, sxz, sxy]
    ndim : int
        Spatial dimension, either 2 or 3.

    Returns
    -------
    stress_vm : np.ndarray, shape (nels, 1)
        Element-wise von Mises stress.
    """
    if ndim == 2:
        sxx = stress[:, 0]
        syy = stress[:, 1]
        sxy = stress[:, 2]
        stress_vm = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)[:, None]
    else:
        sxx = stress[:, 0]
        syy = stress[:, 1]
        szz = stress[:, 2]
        syz = stress[:, 3]
        sxz = stress[:, 4]
        sxy = stress[:, 5]
        stress_vm = np.sqrt(
            0.5 * ((sxx - syy)**2 +
                   (syy - szz)**2 +
                   (szz - sxx)**2) +
            3.0 * (syz**2 + sxz**2 + sxy**2)
        )[:, None]
    return stress_vm

def dsvm_ds(stress: np.ndarray,
                       stress_vm: np.ndarray,
                       ndim: int,
                       vm_floor: float = 1e-14) -> np.ndarray:
    """
    Compute the derivative of von Mises stress with respect to the
    stress components in Voigt notation.

    Parameters
    ----------
    stress : np.ndarray, shape (nels, 3) in 2D or (nels, 6) in 3D
        Stress in Voigt notation.
    stress_vm : np.ndarray, shape (nels, 1)
        Element-wise von Mises stress.
    ndim : int
        Spatial dimension, either 2 or 3.
    vm_floor : float
        Small lower bound to avoid division by zero.

    Returns
    -------
    dsvm : np.ndarray, same shape as stress
        Derivative of von Mises stress with respect to stress.
    """
    vm_safe = np.maximum(stress_vm[:, 0], vm_floor)
    dsvm = np.zeros_like(stress)

    if ndim == 2:
        sxx = stress[:, 0]
        syy = stress[:, 1]
        sxy = stress[:, 2]
        dsvm[:, 0] = (2.0 * sxx - syy) / (2.0 * vm_safe)
        dsvm[:, 1] = (2.0 * syy - sxx) / (2.0 * vm_safe)
        dsvm[:, 2] = 3.0 * sxy / vm_safe

    else:
        sxx = stress[:, 0]
        syy = stress[:, 1]
        szz = stress[:, 2]
        syz = stress[:, 3]
        sxz = stress[:, 4]
        sxy = stress[:, 5]
        dsvm[:, 0] = (2.0 * sxx - syy - szz) / (2.0 * vm_safe)
        dsvm[:, 1] = (2.0 * syy - szz - sxx) / (2.0 * vm_safe)
        dsvm[:, 2] = (2.0 * szz - sxx - syy) / (2.0 * vm_safe)
        dsvm[:, 3] = 3.0 * syz / vm_safe
        dsvm[:, 4] = 3.0 * sxz / vm_safe
        dsvm[:, 5] = 3.0 * sxy / vm_safe
    return dsvm