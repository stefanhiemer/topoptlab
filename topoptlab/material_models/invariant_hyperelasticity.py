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
    Compute isotropic, transversal isotropic or orthotropic invariants of matrix M.

    Invariant order:
        no A, no B : I1, I2, I3
        A only     : I1, I2, I3, I4, I5
        A and B    : I1, I2, I3, I4, I5, I6, I7

    where
        I1 = tr(M)
        I2 = 1/2 M:M
        I3 = det(M)
        I4 = A:M
        I5 = A:(M @ M)
        I6 = B:M
        I7 = B:(M @ M)

    Parameters
    ----------
    M : np.ndarray
        matrix for which to calculate invariants.
    A : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    B : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.

    Returns
    -------
    invariants : ndarray
        isotropic invariants shape (..., 3) if A and B are omitted.
        orthotropic invariants shape (..., 7) if A and B are provided.
    """
    #
    if A is None and B is not None:
        raise ValueError("A cannot be None while B is not None.")
    #
    ndim = M.shape[-1]
    batch_shape = M.shape[:-2]
    #
    if A is not None:
        A = _as_structure_tensor(A, batch_shape, ndim, "A")
    if B is not None:
        B = _as_structure_tensor(B, batch_shape, ndim, "B")
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
    # orthotropic
    if B is not None:
        invariants = invariants + [(B*M).sum(axis=(-2,-1)), 
                                   (B*M2).sum(axis=(-2,-1))]
    return np.stack(invariants, 
                    axis=-1)

def invariant_gradients(M: np.ndarray,
                        A: Union[None, np.ndarray] = None,
                        B: Union[None, np.ndarray] = None, 
                        symmetric: bool = True,
                        ) -> np.ndarray:
    """
    First derivatives dI_i/dM.

    Order:
        no A, no B : I1, I2, I3
        A only     : I1, I2, I3, I4, I5
        A and B    : I1, I2, I3, I4, I5, I6, I7
    
    where
        I1 = tr(M)
        I2 = 1/2 M:M
        I3 = det(M)
        I4 = A:M
        I5 = A:(M @ M)
        I6 = B:M
        I7 = B:(M @ M)

    Parameters
    ----------
    M : np.ndarray
        matrix for which to calculate invariants.
    A : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    B : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    symmetric : bool
        if True, M is assumed to be symmetric.

    Returns
    -------
    invariants_dM : ndarray
        derivatives of isotropic invariants shape (..., 3)+M.shape if A and B are omitted.
        derivatives of orthotropic invariants shape (..., 7) if A and B are provided.
    """
    if A is None and B is not None:
        raise ValueError("A cannot be None while B is not None.")
    #
    ndim = M.shape[-1]
    batch_shape = M.shape[:-2]
    #
    if A is not None:
        A = _as_structure_tensor(A, batch_shape, ndim, "A")
    if B is not None:
        B = _as_structure_tensor(B, batch_shape, ndim, "B")
    #
    gradients = [np.broadcast_to(np.eye(ndim).reshape((1,)*len(batch_shape)+(ndim,ndim)), 
                                 M.shape), # d tr(M) / dM
                 M, # d 1/2 M:M / dM
                 np.linalg.det(M)[..., None, None]*safe_inverse(M).swapaxes(-1, -2)] # d det(M) / dM
    #
    if A is not None:
        if symmetric:
            gradients += [A, # d A:M / dM
                          A @ M + M @ A] # d A:(M @ M) / dM
        else:
            gradients += [A, # d A:M / dM
                          A @ M.swapaxes(-1, -2) + A.swapaxes(-1, -2) @ M] # d A:(M @ M) / dM
    #
    if B is not None:
        if symmetric:
            gradients += [B, # d B:M / dM
                          B @ M + M @ B] # d B:(M @ M) / dM
        else:
            gradients += [B, # d B:M / dM
                          B @ M.swapaxes(-1, -2) + B.swapaxes(-1, -2) @ M] # d B:(M @ M) / dM
    #
    return np.stack(gradients, axis=-3)

def invariant_hessian(M: np.ndarray,
                      A: Union[None, np.ndarray] = None,
                      B: Union[None, np.ndarray] = None,
                      symmetric: bool = True,
                      ) -> np.ndarray:
    """
    Second derivatives dI_i/dM.

    Order:
        no A, no B : I1, I2, I3
        A only     : I1, I2, I3, I4, I5
        A and B    : I1, I2, I3, I4, I5, I6, I7
    
    where
        I1 = tr(M)
        I2 = 1/2 M:M
        I3 = det(M)
        I4 = A:M
        I5 = A:(M @ M)
        I6 = B:M
        I7 = B:(M @ M)

    Parameters
    ----------
    M : np.ndarray
        matrix for which to calculate invariants.
    A : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    B : None or np.ndarray
        Union orthotropic structure tensors. Omit both for isotropic
        invariants I1, I2, I3. Provide both for the 7-invariant orthotropic
        basis.
    symmteric : bool
        if True, M is assumed to be symmetric.

    Returns
    -------
    invariants_dM2 : ndarray
        2nd derivatives of isotropic invariants shape (..., 3)+M.shape if A and B are omitted.
        2nd derivatives of orthotropic invariants shape (..., 7) if A and B are provided.
    """
    if not symmetric:
        raise NotImplementedError("Voigt Hessian is currently intended for symmetric M.")
    if A is None and B is not None:
        raise ValueError("A cannot be None while B is not None.")
    #
    ndim = M.shape[-1]
    batch_shape = M.shape[:-2]
    nvoigt = ndim * (ndim + 1) // 2
    #
    if A is not None:
        A = _as_structure_tensor(A, batch_shape, ndim, "A")
    if B is not None:
        B = _as_structure_tensor(B, batch_shape, ndim, "B")
    #
    zero = np.zeros(batch_shape + (nvoigt, nvoigt), dtype=M.dtype)
    #
    H = []
    # isotropic invariants
    H.append(zero.copy()) # I1 = tr(M)
    H.append(np.broadcast_to(np.eye(nvoigt, dtype=M.dtype), 
             batch_shape + (nvoigt, nvoigt))) # I2 = 1/2 M:M
    H.append(_det_hessian_voigt(M)) # I3 = det(M)

    if A is not None:
        # I4 = A:M
        H.append(zero.copy())

        # I5 = A:(M @ M)
        H.append(_quad_structure_hessian_voigt(A))

    if B is not None:
        # I6 = B:M
        H.append(zero.copy())

        # I7 = B:(M @ M)
        H.append(_quad_structure_hessian_voigt(B))

    return np.stack(H, axis=-3)

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

def stress_2pk_invariants(F: np.ndarray,
                          dpsi_dI: Callable,
                          A: Union[None, np.ndarray] = None,
                          B: Union[None, np.ndarray] = None,
                          E: Union[None, np.ndarray] = None,
                          I: Union[None, np.ndarray] = None,
                          matconst: Dict = {},
                          **kwargs: Any) -> np.ndarray:
    """
    Return 2PK stress and material tangent for a generic invariant law.
    
    If I is provided, the invariants are not recomputed.
    If matconst is provided, it is merged with kwargs and passed to dpsi_dI.
    
    Parameters
    ----------
    F : np.ndarray
        Deformation gradient, shape (..., ndim, ndim).
    dpsi_dI : callable
        Callable returning dPsi/dI, shape (..., n_inv).
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
    """
    # compute strain measures
    if E is None:
        E = lagrangian_strain(F=F)
    if I is None:
        I = invariants(M=E, A=A, B=B)
    # 
    dW = dpsi_dI(I, **matconst)
    #
    dIdE = invariant_gradients(M=E,
                               A=A,
                               B=B,
                               symmetric=True)
    #
    if dW.shape[-1] != dIdE.shape[-3]:
        raise ValueError(f"dpsi_dI returned {dW.shape[-1]} derivatives, "
                         f"but the invariant basis has {dIdE.shape[-3]} invariants.")

    return to_voigt(np.einsum("...a,...aij->...ij", 
                              dW, 
                              dIdE))

def consttensor_2pk_invariants(F: np.ndarray,
                               dpsi_dI: Callable,
                               d2psi_dIdI: Callable,
                               E: Union[None,np.ndarray] = None,
                               I: Union[None,np.ndarray] = None,
                               matconst: Union[None,Dict] = None,
                               **kwargs: Any,
                               ) -> np.ndarray:
    """
    Material tangent dS/dE in Voigt notation from invariant derivatives..

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
    c : np.ndarray
        Material tangent dS/dE in Voigt notation,
        shape (..., ndim*(ndim+1)//2, ndim*(ndim+1)//2).
    """
    if E is None:
        E = lagrangian_strain(F)
    if I is None:
        I = invariants_E(E)
    #
    if matconst is not None:
        matconst = {}
    #
    dW = dpsi_dI(I, **matconst)
    ddW = d2psi_dIdI(I, **matconst)
    #
    dIdm = to_voigt(invariant_gradients_E(E))
    d2Idm2 = invariant_hessian_E(E)
    #
    return (np.einsum("...ab,...ai,...bj->...ij", ddW, dIdm, dIdm)
            + np.einsum("...a,...aij->...ij", dW, d2Idm2))

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
