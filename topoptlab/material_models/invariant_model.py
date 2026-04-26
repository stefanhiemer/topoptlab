# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Tuple,Union

import numpy as np

from topoptlab.utils import safe_inverse

def invariant_matmodel(F: np.ndarray,
                       dpsi_dI: Callable,
                       d2psi_dIdI: Callable,
                       A: Union[None,np.ndarray] = None,
                       B: Union[None,np.ndarray] = None,
                       **kwargs: Any,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2PK stress and material tangent for a generic invariant law.

    Parameters
    ----------
    F : np.ndarray
        Deformation gradient, shape (..., ndim, ndim).
    dpsi_dI : callable
        Callable returning dPsi/dI, shape (..., n_inv).
    d2psi_dIdI : callable
        Callable returning d2Psi/dI dI, shape (..., n_inv, n_inv).
    A : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    B : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    **kwargs
        Extra material parameters passed to dpsi_dI and d2psi_dIdI or simple
        trash collector.

    Returns
    -------
    s
        2PK stress in Voigt notation, shape (..., ndim*(ndim+1)//2).
    c
        Material tangent dS/dE in Voigt notation,
        shape (..., ndim*(ndim+1)//2, ndim*(ndim+1)//2).
    """
    s = stress_2pk_invariants(F=F,
                              C=C,
                              dpsi_dI=dpsi_dI,
                              A=A,
                              B=B,
                              **kwargs)
    c = consttensor_2pk_invariants(F=F,
                                   C=C,
                                   dpsi_dI=dpsi_dI,
                                   d2psi_dIdI=d2psi_dIdI,
                                   A=A,
                                   B=B,
                                   **kwargs)
    return s, c
