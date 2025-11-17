# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Dict, List, Tuple
from functools import partial 

import numpy as np
from scipy.sparse import sparray
from scipy.sparse.linalg import spsolve, LinearOperator

from topoptlab.linear_solvers import smoothed_jacobi, res_norm

def apply_multigrid(b : np.ndarray, 
                    A : sparray, x0 : np.ndarray,
                    interpolators : List,
                    cycle : Callable, tol: float,
                    smoother_fnc : Callable,
                    smoother_kws : Dict,
                    max_cycles : int = 1,
                    nlevels : int = 2,
                    conv_criterium: Callable = res_norm,
                    conv_args: Dict = {}) -> Tuple[np.ndarray,int]:
    """
    Apply a generic multigrid solver for the linear problem Ax=b. In this
    function we assume that the interpolation from coarse to fine grid
    via the interpolator P also gives us the map from fine to coarse grid via 
    the transpose of the prolongator P.T. We might call P.T a 
    restrictor/coarsener.

    Parameters
    ----------
    
    b : np.ndarray
        right hand side of full system.
    A : scipy.sparse.sparray
        system matrix (e. g. stiffness matrix)
    x0 : None or np.ndarray
        initial guess for solution. 
    interpolators : list
        list of matrices P that interpolate from coarse to fine grid. Must be
        initialized before calling this function.
    cycle : callable
        a multigrid cycle. Currently only V-cycle is
        available, but the common versions are V,F and W.
    tol : float
        convergence tolerance.
    smoother_fnc : callable
        function for smoothing the error (e. g. Gauss-Seidel or Jacobi
        iteration).
    smoother_kws : dict
        keywords for the smoother.
    max_cycles : int
        maximum number of cycles.
    nlevels : int
        number of grid levels.
    conv_criterium : callable
        converge criterion.
    conv_args : dict
        arguments regarding the convergence criterion.

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
    #
    r = b - A@x
    for i in np.arange(max_cycles):
        # check convergence
        if conv_criterium(r=r,atol=tol,**conv_args):
            break
        # one
        x[:],info_cycle = cycle(A=A,b=b,x0=x,
                                lvl=0,
                                interpolators=interpolators,
                                smoother_fnc=smoother_fnc,
                                smoother_kws=smoother_kws,
                                nlevels=nlevels)
        # residual
        r[:] = b - A @ x
        print(np.abs(r).max())
    return x

def vcycle(A : sparray, b : np.ndarray, x0 : np.ndarray,
           lvl : int,
           interpolators : List,
           smoother_fnc : Callable,
           smoother_kws : Dict,
           nlevels : int) -> Tuple[np.ndarray,int]:
    """
    Generic, single recursive V-cycle iteration to solve the linear problem
    Ax=b. In the current implementation we assume that the interpolation from
    coarse to fine grid via the prolongator/interpolator P also gives us the
    map from fine to coarse grid via the transpose of the prolongator P.T which
    we might call the restrictor/coarsener.

    Parameters
    ----------
    A : scipy.sparse.sparray
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
    smoother_fnc : callable
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
    x,info_smoother = smoother_fnc(A=A,b=b,x0=x0,
                                   **smoother_kws)
    # compute residual
    r = b - A@x
    # compute coarse grid correction
    if lvl == nlevels-2:
        xc = spsolve(interpolators[lvl].T@A@interpolators[lvl],
                     interpolators[lvl].T@r)
    else:
        xc,info_vcycle = vcycle(A=interpolators[lvl].T@A@interpolators[lvl], 
                                b=interpolators[lvl].T@r, 
                                x0=interpolators[lvl].T@x,
                                lvl=lvl+1, 
                                interpolators=interpolators,
                                smoother_fnc=smoother_fnc, 
                                smoother_kws=smoother_kws,
                                nlevels=nlevels)
    # interpolate
    x = x + interpolators[lvl]@xc
    # post-smooth
    x,info = smoother_fnc(A,b,x0=x,**smoother_kws)
    return x,info

def multigrid_preconditioner(A: sparray, 
                             b: np.ndarray, 
                             x0: np.ndarray,
                             create_interpolators: Callable,
                             interpolator_kw: Dict,
                             cycle: Callable = vcycle, 
                             tol: float = 1e-6,
                             smoother_fnc: Callable = smoothed_jacobi,
                             smoother_kws: Dict = {},
                             max_cycles: int = 1) -> LinearOperator:
    """
    Generic multigrid preconditioner for the linear problem Ax=b. In the current
    implementation we assume that the interpolation from coarse to fine grid
    via the interpolator P also gives us the map from fine to
    coarse grid via the transpose of the prolongator P.T. We might call P.T
    a restrictor/coarsener.

    Parameters
    ----------
    A : scipy.sparse.sparray
        system matrix (e. g. stiffness matrix)
    b : np.ndarray
        right hand side of full system.
    x0 : np.ndarray
        initial guess for solution.
    create_interpolators : Callable
        create list of matrices P that interpolate from coarse to fine grid. 
    interpolator_kw : dict
        keywords needed to construct the interpolators.
    cycle : callable
        a multigrid cycle. Currently only V-cycle is
        available, but the common versions are V,F and W.
    tol : float
        convergence tolerance.
    smoother_fnc : callable
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
    M : scipy.sparse.linalg.LinearOperator
        multigrid preconditioner

    """
    #
    interpolators = create_interpolators(A=A, **interpolator_kw)
    #
    nlevels = len(interpolators)+1
    #
    matvec = partial(apply_multigrid,
                     A=A, x0=np.zeros(A.shape[0]),
                     interpolators=interpolators,
                     cycle=cycle, 
                     tol=tol,
                     smoother_fnc=smoother_fnc,
                     smoother_kws=smoother_kws,
                     max_cycles=max_cycles,
                     nlevels=nlevels)
    #
    return LinearOperator(A.shape, 
                          matvec=matvec,
                          dtype=A.dtype)
    
    