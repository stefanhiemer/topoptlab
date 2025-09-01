from typing import Any,Union,Tuple

import numpy as np
from scipy.sparse import issparse,diags,triu,tril,csc_array
from scipy.sparse.linalg import spsolve_triangular

def gauss_seidel(A: csc_array, b: np.ndarray,
                 x0: Union[None,np.ndarray] = None, 
                 tol: float = 1e-8, max_iter: int = 1000,
                 L: Union[None,csc_array] = None, 
                 U: Union[None,csc_array] = None,
                 **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Gauss-Seidel solver for Ax = b. We rewrite 
    
    L x^i = omega b - U x^(i-1)
    
    to avoid Python for loops.
    
    Parameters
    ----------
    A : scipy.sparse.csc_array
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    tol : float
        convergence tolerance.
    maxiter : int
        maximum number of iterations.
    L : scipy.sparse.csc_array or None
        lower triangular matrix of A (with diagonal).
    U : scipy.sparse.csc_array or None
        upper triangular matrix of A (without diagonal).

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged, info>0: exited due to reaching maximum number of 
        iterations.
    """ 
    # type check
    if not issparse(A):
        raise TypeError("Matrix A must be a sparse array.")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy ndarray.")
    # initial guess
    if x0 is None:
        x = np.zeros(b.shape)
    else:
        x = x0.copy()
    # inverse of diagonal
    if L is None and U is None:
        L = tril(A,k=0,format="csc")
        U =  triu(A,k=1,format="csc")
    #
    r = np.zeros(x.shape,dtype=np.float64)
    for i in np.arange(max_iter):
        # SRO update
        x[:] = spsolve_triangular(L, b - U@x,
                                  lower=True)
        # residual
        r[:] = b - A @ x
        # check convergence
        if np.abs(r).max() < tol:
            i = 0
            break
    return x, i

def smoothed_jacobi(A: csc_array, b: np.ndarray, 
                    x0: Union[None,np.ndarray] = None, 
                    omega: float = 0.67, 
                    tol: float = 1e-8, max_iter: int = 1000,
                    **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Smoothed Jacobi iterative solver for Ax = b.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    tol : float
        convergence tolerance
    maxiter : int
        maximum number of iterations

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged, info>0: exited due to reaching maximum number of 
        iterations.
    """ 
    # type check
    if not issparse(A):
        raise TypeError("Matrix A must be a sparse matrix.")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy ndarray.")
    # initial guess
    if x0 is None:
        x = np.zeros(b.shape)
    else:
        x = x0.copy()
    # inverse of diagonal
    Dinv = 1. / A.diagonal()
    # initialize residual
    r = np.zeros(x.shape,dtype=np.float64)
    for i in np.arange(max_iter):
        # smoothed Jacobi update
        x[:] = x + omega * ( Dinv*b - Dinv*(A@x) )
        # residual
        r[:] = b - A @ x
        # check convergence
        if np.abs(r).max() < tol:
            i = 0
            break
    return x, i

def modified_richardson(A: csc_array, b: np.ndarray, 
                        x0: Union[None,np.ndarray] = None, 
                        omega: float = 0.1, 
                        tol: float = 1e-8, max_iter: int = 1000,
                        **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Modified Richardson iterative solver for Ax = b.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    tol : float
        convergence tolerance
    maxiter : int
        maximum number of iterations

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged, info>0: exited due to reaching maximum number of 
        iterations.
    """ 
    # type check
    if not issparse(A):
        raise TypeError("Matrix A must be a sparse matrix.")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy ndarray.")
    # initial guess
    if x0 is None:
        x = np.zeros(b.shape)
    else:
        x = x0.copy()
    # initial residual
    r = b - A @ x
    for i in np.arange(max_iter):
        # richardson update
        x[:] = x + omega*r
        # residual
        r[:] = b - A @ x
        # check convergence
        if np.abs(r).max() < tol:
            i = 0
            break
    return x, i

def successive_overrelaxation(A: csc_array, b: np.ndarray, 
                              x0: Union[None,np.ndarray] = None, 
                              omega: float = 0.5, 
                              tol: float = 1e-8, max_iter: int = 1000,
                              D: Union[None,csc_array] = None, 
                              A_u: Union[None,csc_array] = None, 
                              A_l: Union[None,csc_array] = None,
                              **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Successive over-relaxation (SRO) solver for Ax = b. We rewrite 
    
    (D+omega L) x^i = omega b - [ omega U + (omega - 1)D ]x^(i-1)
    
    to
    
    A_l x^i = omega b - A_u x^i-1
    
    to avoid repeated addition etc. in the sparse matrices.
    
    Parameters
    ----------
    A : scipy.sparse.csc_array
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    tol : float
        convergence tolerance.
    maxiter : int
        maximum number of iterations.
    D : scipy.sparse.csc_array or None
        diagonal of A.
    M_u : scipy.sparse.csc_array or None
        helper matrix (see equation above).
    M_l : scipy.sparse.csc_array or None
        helper matrix (see equation above).

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged, info>0: exited due to reaching maximum number of 
        iterations.
    """ 
    # type check
    if not issparse(A):
        raise TypeError("Matrix A must be a sparse array.")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy ndarray.")
    # initial guess
    if x0 is None:
        x = np.zeros(b.shape)
    else:
        x = x0.copy()
    # inverse of diagonal
    if D is None:
        D = diags(A.diagonal(),format="csc")
        M_l = D + omega * tril(A,k=-1,format="csc")
        M_u =  omega * triu(A,k=1,format="csc") + (omega - 1) * D
    #
    r = np.zeros(x.shape,dtype=np.float64)
    for i in np.arange(max_iter):
        # SRO update
        x[:] = spsolve_triangular(M_l, omega * b - M_u@x,
                                  lower=True)
        # residual
        r[:] = b - A @ x
        # check convergence
        if np.abs(r).max() < tol:
            i = 0
            break
    return x, i