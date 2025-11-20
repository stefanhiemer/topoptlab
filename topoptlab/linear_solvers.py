# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Dict,Union,Tuple

import numpy as np
from scipy.sparse import issparse,diags,triu,tril,sparray, coo_array,bsr_array
from scipy.sparse.linalg import spsolve_triangular, LinearOperator
from scipy.sparse.linalg._isolve.iterative import _get_atol_rtol

from topoptlab.log_utils import EmptyLogger,SimpleLogger

def max_res(r: np.ndarray, atol: float,
            **kwargs: Any) -> bool:
    """
    Check if maximum residual smaller than tolerance. This is a very (!) strict
    convergence criterion so should only be used for testing or similar things.
    
    Parameters
    ----------
    r :np.ndarray
        residual.
    atol : float 
        absolute tolerance.

    Returns
    -------
    converged : bool
        if True, max. residual smaller than tolerance. 
    """ 
    return np.abs(r).max() < atol

def res_norm(r: np.ndarray, atol: float,
            **kwargs: Any) -> bool:
    """
    Check if maximum residual smaller than tolerance.
    
    Parameters
    ----------
    r :np.ndarray
        residual.
    tol : float 
        tolerance.

    Returns
    -------
    converged : bool
        if True, max. residual smaller than tolerance. 
    """ 
    return np.linalg.norm(r) < atol

def pcg(A: sparray, b: np.ndarray,
        P: Union[sparray,LinearOperator],
        x0: Union[None,np.ndarray] = None,
        rtol: Union[None,float] = 1e-5,
        atol: Union[None,float] = 0., 
        maxiter: int = 100000,
        conv_criterium: Callable = res_norm,
        conv_args: Dict = {},
        logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
        **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Preconditioned conjugate gradient solver for `Ax=b`, for a symmetric, 
    positive-definite matrix `A`. Iterate until convergence criteria met or the 
    maximum number of iterations is exceeded.

    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    P : scipy.sparse.sparray or scipy.sparse.LinearOperator
        preconditioner that is called at each cg iteration.
    x0 : np.ndarray
        initial guess for solution.
    rtol : None or float
        relative convergence tolerance.
    atol : None or float
        absolute convergence tolerance.
    maxiter : int
        maximum number of iterations.
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.
        
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
    #
    atol,_ = _get_atol_rtol(name="pcg", 
                          b_norm=np.linalg.norm(b), 
                          atol=atol, rtol=rtol)
    # updates of residual and x
    dx = np.zeros(x.shape)
    dr,z = dx.copy(),dx.copy()
    #
    r = b - A@x
    for i in range(maxiter):
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"PCGit. {i}")
            i = 0
            break
        # preconditioner
        z[:] = P@r
        rho_cur = np.dot(r, z)
        if i > 0:
            beta = rho_cur / rho_prev
            dx[:] = dx*beta + z
        else:
            dx[:] = z.copy()
        #
        dr[:] = A@dx
        alpha = rho_cur / np.dot(dx, dr)
        x[:] += alpha*dx
        r[:] -= alpha*dr
        rho_prev = rho_cur
        
    return x, i

def cg(A: sparray, b: np.ndarray,
       x0: Union[None,np.ndarray] = None,
       rtol: Union[None,float] = 1e-5,
       atol: Union[None,float] = 0., 
       maxiter: int = 100000,
       conv_criterium: Callable = res_norm,
       conv_args: Dict = {},
       logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
       **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Conjugate gradient solver for `Ax=b`, for a symmetric, positive-definite 
    matrix `A` without any preconditioning. Iterate until convergence criteria 
    met or the maximum number of iterations is exceeded.
    
    This is purely for teaching and 
    benchmarking reasons and should never be used for any production runs. 
    

    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    rtol : None or float
        relative convergence tolerance.
    atol : None or float
        absolute convergence tolerance.
    maxiter : int
        maximum number of iterations.
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.
        
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
    #
    atol,_ = _get_atol_rtol(name="cg", 
                          b_norm=np.linalg.norm(b), 
                          atol=atol, rtol=rtol)
    # updates of residual and x
    dx = np.zeros(x.shape)
    dr,z = dx.copy(),dx.copy()
    #
    r = b - A@x
    for i in range(maxiter):
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"CGit. {i}")
            i = 0
            break
        # preconditioner
        rho_cur = np.dot(r, r)
        if i > 0:
            beta = rho_cur / rho_prev
            dx[:] = dx*beta + r
        else:
            dx[:] = r.copy()
        #
        dr[:] = A@dx
        alpha = rho_cur / np.dot(dx, dr)
        x[:] += alpha*dx
        r[:] -= alpha*dr
        rho_prev = rho_cur
        
    return x, i


def gauss_seidel(A: sparray, b: np.ndarray,
                 x0: Union[None,np.ndarray] = None, 
                 atol: float = 1e-8, max_iter: int = 1000,
                 L: Union[None,sparray] = None, 
                 U: Union[None,sparray] = None,
                 conv_criterium: Callable = res_norm,
                 conv_args: Dict = {},
                 logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                 **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Gauss-Seidel solver for Ax = b. We re-write each iteration as with the 
    lower and upper triangular matrices L,U 
    
    L x^i = omega b - U x^(i-1)
    
    to avoid using for loops. Iterate until the residual `r=b-Ax` fulfills
    the convergence criteria or the maximum number of iterations is exceeded.
    
    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    atol : float
        abs. convergence tolerance.
    maxiter : int
        maximum number of iterations.
    L : scipy.sparse.sparray or None
        lower triangular matrix of A (with diagonal).
    U : scipy.sparse.sparray or None
        upper triangular matrix of A (without diagonal).
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.

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
        U =  triu(A,k=1,format="csr")
    # initial residual
    r = b - A @ x
    for i in np.arange(max_iter):
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"Gauss-Seidel it. {i}")
            i = 0
            break
        # SRO update
        x[:] = spsolve_triangular(L, b - U@x,
                                  lower=True)
        # residual
        r[:] = b - A @ x
        
    return x, i

def smoothed_jacobi(A: sparray, b: np.ndarray, 
                    x0: Union[None,np.ndarray] = None, 
                    omega: float = 0.67, 
                    atol: float = 1e-8, max_iter: int = 1000,
                    conv_criterium: Callable = res_norm,
                    conv_args: Dict = {},
                    logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                    **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Smoothed Jacobi iterative solver for `Ax = b`. Iterate until the residual 
    `r=b-Ax` fulfills r.max()<tol or the maximum number of iterations is 
    exceeded.

    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    atol : float
        convergence tolerance.
    maxiter : int
        maximum number of iterations.
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.

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
    # initial residual
    r = b - A @ x
    for i in np.arange(max_iter):
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"Smoothed-Jacobi it. {i}")
            i = 0
            break
        # smoothed Jacobi update
        x[:] = x + omega * ( Dinv*b - Dinv*(A@x) )
        # residual
        r[:] = b - A @ x
    return x, i

def modified_richardson(A: sparray, b: np.ndarray, 
                        x0: Union[None,np.ndarray] = None, 
                        omega: float = 0.1, 
                        atol: float = 1e-8, max_iter: int = 1000,
                        conv_criterium: Callable = res_norm,
                        conv_args: Dict = {},
                        logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                        **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Modified Richardson iterative solver for `Ax=b`. Iterate until the residual 
    `r=b-Ax` fulfills r.max()<tol or the maximum number of iterations is 
    exceeded.

    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    atol : float
        abs. convergence tolerance.
    maxiter : int
        maximum number of iterations.
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.

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
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"modified Richardson it. {i}")
            i = 0
            break
        # richardson update
        x[:] = x + omega*r
        # residual
        r[:] = b - A @ x
    return x, i

def successive_overrelaxation(A: sparray, b: np.ndarray, 
                       x0: Union[None,np.ndarray] = None, 
                       omega: float = 0.5, 
                       atol: float = 1e-8, 
                       max_iter: int = 1000,
                       D: Union[None,sparray] = None, 
                       A_u: Union[None,sparray] = None, 
                       A_l: Union[None,sparray] = None,
                       conv_criterium: Callable = res_norm,
                       conv_args: Dict = {},
                       logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                       **kwargs: Any) -> Tuple[np.ndarray,int]:
    """
    Successive over-relaxation (SRO) solver for `Ax=b`. We rewrite 
    
    (D+omega L) x^i = omega b - [ omega U + (omega - 1)D ]x^(i-1)
    
    to
    
    A_l x^i = omega b - A_u x^i-1
    
    to avoid repeated addition etc. in the sparse matrices. Iterate until the 
    residual `r=b-Ax` fulfills r.max()<tol or the maximum number of iterations 
    is exceeded.
    
    Parameters
    ----------
    A : scipy.sparse.sparray
        matrix of linear system.
    b : np.ndarray 
        right hand side of linear system.
    x0 : np.ndarray
        initial guess for solution.
    omega : float
        damping factor usually between 0/1.
    atol : float
        abs. convergence tolerance.
    maxiter : int
        maximum number of iterations.
    D : scipy.sparse.sparray or None
        diagonal of A.
    M_u : scipy.sparse.sparray or None
        helper matrix (see equation above).
    M_l : scipy.sparse.sparray or None
        helper matrix (see equation above).
    conv_criterium : callable
        convergence criterium.
    conv_args : dict
        additional arguments for convergence criterium.
    logger : BaseLogger
        logger object to log performance.

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
    # initial residual
    r = b - A @ x
    for i in np.arange(max_iter):
        # check convergence
        if conv_criterium(r=r,atol=atol,**conv_args):
            logger.perf(f"successive overrelaxation it. {i}")
            i = 0
            break
        # SRO update
        x[:] = spsolve_triangular(M_l, omega * b - M_u@x,
                                  lower=True)
        # residual
        r[:] = b - A @ x
    return x, i