# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Tuple,Union

import numpy as np

from topoptlab.elements.strain_measures import lagrangian_strain
from topoptlab.utils import safe_inverse
from topoptlab.voigt import to_voigt

def _as_structure_tensor(A: Union[None, np.ndarray],
                         batch_shape: Tuple[int, ...],
                         ndim: int,
                         name: str) -> Union[None, np.ndarray]:
    if A is None:
        return None
    A = np.asarray(A)
    if A.shape[-2:] != (ndim, ndim):
        raise ValueError(f"{name} must have trailing shape ({ndim}, {ndim}), got {A.shape}.")
    return np.broadcast_to(A, 
                           batch_shape + (ndim, ndim))

def invariants(M: np.ndarray,
               A: Union[None,np.ndarray] = None,
               B: Union[None,np.ndarray] = None,
               ) -> np.ndarray:
    """
    Compute isotropic or orthotropic invariants of matrix M.

    Returns
    -------
    invariants : ndarray
        isotropic invariants shape (..., 3) if A and B are omitted.
        orthotropic invariants shape (..., 7) if A and B are provided.
    """
    #
    A = _as_structure_tensor(A, A.shape[:-2], M.shape[-1], "A")
    B = _as_structure_tensor(B, B.shape[:-2], M.shape[-1], "B")
    # isotropic invariants
    M2 = M**2
    invariants = [np.trace(M, axis1=-2, axis2=-1), 
                  0.5 * M2.sum(axis=(-2,-1)), 
                  np.linalg.det(M)]
    # transversal isotropic invariants
    if A is not None:
        M2 = M@M
        invariants = invariants + [(A*M).sum(axis=(-2,-1)), 
                                   (A*M2).sum(axis=(-2,-1))]
    #
    if A is None and B is not None:
        raise ValueError("A cannot be None while B is not None.")
    # orthotropic
    else:
        invariants = invariants + [(B*M).sum(axis=(-2,-1)), 
                                   (B*M2).sum(axis=(-2,-1))]
    return np.stack(invariants, 
                    axis=-1)

def invariant_gradients(M: np.ndarray,
                        A: Union[None, np.ndarray] = None,
                        B: Union[None, np.ndarray] = None) -> np.ndarray:
    #
    A = _as_structure_tensor(A, A.shape[:-2], M.shape[-1], "A")
    B = _as_structure_tensor(B, B.shape[:-2], M.shape[-1], "B")
    #
    I = _identity(batch_shape, ndim).astype(M.dtype, copy=False)
    grad = [np.broadcast_to(I, M.shape),
            M,
            _cofactor(M)]

    if A is None and B is None:
        return np.stack(grad, axis=-3)
    if A is None and B is not None:
        raise ValueError("A cannot be None while B is not None.")

    grad += [A,
             A @ M + M @ A]

    if B is not None:
        grad += [B,
                 B @ M + M @ B]

    return np.stack(grad, axis=-3)

def eng_density_invariants(F: np.ndarray,
                           psi: Callable,
                           A: Union[None, np.ndarray] = None,
                           B: Union[None, np.ndarray] = None,
                           E: Union[None, np.ndarray] = None,
                           **kwargs: Any) -> np.ndarray:
    
    if E is None:
        E = lagrangian_strain(F=F)
    I = invariants(M=E, 
                   A=A, 
                   B=B)
    return psi(I, **kwargs)

def stress_2pk_invariants(F : np.ndarray, 
                          dpsi_dI: Callable,
                          d2psi_dIdI: Callable,
                          A : Union[None,np.ndarray],
                          B : Union[None,np.ndarray],
                          E : Union[None,np.ndarray] = None, 
                          **kwargs) -> np.ndarray:
    if E is None:
        E = lagrangian_strain(F=F)
    I = invariants(M=E,
                   A=A,
                   B=B)
    dW = dpsi_dI(I, 
                 **kwargs)
    dIdM = invariant_gradients(M=E,
                               A=A, 
                               B=B)

    if dW.shape[-1] != dIdM.shape[-3]:
        raise ValueError(f"dpsi_dI returned {dW.shape[-1]} derivatives, "
                         f"but the invariant basis has {dIdM.shape[-3]} invariants.")

    S = np.einsum("...a,...aij->...ij", dW, dIdM)
    return to_voigt(S)

def invariant_matmodel(F: np.ndarray,
                       dpsi_dI: Callable,
                       d2psi_dIdI: Callable,
                       A : Union[None,np.ndarray] = None,
                       B : Union[None,np.ndarray] = None,
                       E : Union[None,np.ndarray()] = None,
                       **kwargs : Any,
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
    E : None or np.ndarray
        Green Lagrange strain of shape (...,ndim,ndim).
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
    #
    if E is None:
        E = lagrangian_strain(F=F)
    #
    s = stress_2pk_invariants(F=F,
                              C=C,
                              dpsi_dI=dpsi_dI,
                              A=A,
                              B=B,
                              **kwargs)
    #
    c = consttensor_2pk_invariants(F=F,
                                   C=C,
                                   dpsi_dI=dpsi_dI,
                                   d2psi_dIdI=d2psi_dIdI,
                                   A=A,
                                   B=B,
                                   **kwargs)
    return s, c
