import numpy as np
from scipy.sparse import issparse
import scipy.sparse.linalg as spla

def smoothed_jacobi(A, b, x0=None, omega=0.67, tol=1e-8, max_iter=1000):
    """
    Smoothed Jacobi iterative solver for Ax = b.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
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
    if not issparse(A):
        raise ValueError("Matrix A must be a sparse matrix.")
    
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    Dinv = 1.0 / A.diagonal()  # Inverse of diagonal elements
    r = b - A @ x  # Initial residual
    #
    for i in range(max_iter):
        # Smoothed Jacobi update
        x = x[:] + omega * ( Dinv*b - Dinv*(A@x) )
        r = b - A @ x  # Compute new residual
        if np.abs(r).max() < tol:
            i = 0
            break  # Converged
    return x, i

def modified_richardson(A, b, x0=None, omega=0.67, tol=1e-8, max_iter=1000):
    """
    Modified Richardson iterative solver for Ax = b.

    Parameters
    ----------
    A : scipy.sparse.csc_matrix
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
    if not issparse(A):
        raise ValueError("Matrix A must be a sparse matrix.")
    
    n = A.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
        
    r = b - A @ x  # Initial residual
    #
    for i in range(max_iter):
        # Smoothed Jacobi update
        x = x[:] + omega*r
        r = b - A @ x  # Compute new residual
        if np.abs(r).max() < tol:
            i = 0
            break  # Converged
    return x, i