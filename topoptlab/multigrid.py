from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

def multigrid_solver(A: csc_array, b: np.ndarray, x0: np.ndarray,
                     interpolators: List,
                     cycle: Callable, tol: float,
                     smoother: Callable,
                     smoother_kws: Dict,
                     max_cycles: int,
                     nlevels: int) -> Tuple[np.ndarray,int]:
    """
    Generic multigrid solver for the linear problem Ax=b. In the current
    implementation we assume that the interpolation from coarse to fine grid
    via the prolongator/interpolator P also gives us the map from fine to
    coarse grid via the transpose of the prolongator P.T. We might call P.T
    a restrictor/coarsener.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        system matrix (e. g. stiffness matrix)
    b : np.ndarray
        right hand side of full system.
    x0 : np.ndarray
        initial guess for solution.
    interpolators : list
        list of matrices P that interpolate from coarse to fine grid. Must be
        initialized before calling this function.
    cycle : callable
        a multigrid cycle. Currently only V-cycle is
        available, but the common versions are V,F and W.
    tol : float
        convergence tolerance.
    smoother : callable
        function for smoothing the error (e. g. Gauss-Seidel or Jacobi
        iteration).
    smoother_kws : dict
        keywords for the smoother.
    max_cycles : int
        maximum number of cycles.
    nlevels : int
        number of grid levels.

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged in final post-smoothing , info>0: exited during last
        post-smoothing due to maximum number of iterations.

    """
    #
    x = np.zeros(x0.shape)
    r = np.zeros(b.shape)
    for i in np.arange(max_cycles):
        # one
        x[:] = cycle(A=A,b=b,x0=x0,lvl=0,
                     interpolators=interpolators,
                     smoother=smoother,
                     smoother_kws=smoother_kws,
                     nlevels=nlevels)
        # residual
        r[:] = b - A @ x
        # check convergence
        if np.abs(r).max() < tol:
            i = 0
            break
    return x, i

def vcycle(A: csc_array,b: np.ndarray, x0: np.ndarray,
           lvl: int,
           interpolators: List,
           smoother: Callable,
           smoother_kws: Dict,
           nlevels: int) -> Tuple[np.ndarray,int]:
    """
    Generic, single recursive V-cycle iteration to solve the linear problem
    Ax=b. In the current implementation we assume that the interpolation from
    coarse to fine grid via the prolongator/interpolator P also gives us the
    map from fine to coarse grid via the transpose of the prolongator P.T which
    we might call the restrictor/coarsener.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        system matrix (e. g. stiffness matrix)
    b : np.ndarray
        right hand side of full system.
    x0 : np.ndarray
        initial guess for solution.
    lvl : int
        number of current level (maximum is nlevels-1).
    interpolators : list
        list of matrices P that interpolate from coarse to fine grid. Must be
        initialized before calling this function.
    smoother : callable
        function for smoothing the error (e. g. Gauss-Seidel or Jacobi
        iteration).
    smoother_kws : dict
        keywords for the smoother.
    nlevels : int
        number of grid levels.

    Returns
    -------
    x : np.ndarray
        final result for solution.
    info : int
        0: converged in final post-smoothing , info>0: exited during last
        post-smoothing due to maximum number of iterations.

    """
    # pre-smooth
    x,info_smoother = smoother(A=A,b=b,x0=x0,
                               **smoother_kws)
    # compute residual
    r = b - A@x
    # compute coarse grid correction
    if lvl == nlevels-1:
        xc = spsolve( interpolators[lvl].T@r )
    else:
        xc,info_vcycle = vcycle(A=interpolators[lvl].T@A@interpolators[lvl], 
                                b=interpolators[lvl].T@r, 
                                x0=interpolators[lvl].T@x,
                                lvl=lvl+1, interpolators=interpolators,
                                smoother=smoother, smoother_kws=smoother_kws,
                                nlevels=nlevels)
    # interpolate
    x = x + interpolators[lvl]@xc
    # post-smooth
    x,info = smoother(A,b,x0=x,**smoother_kws)
    return x,info
