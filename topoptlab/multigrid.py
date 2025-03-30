import numpy as np
from scipy.sparse.linalg import spsolve

def multigrid_solver(A,b,x0,
                     prolongators,
                     cycle, tol,
                     smoother,
                     smoother_kws,
                     max_cycles,
                     nlevels):
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
    prolongators : list
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
                     prolongators=prolongators,
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

def vcycle(A,b,x0,lvl,
           prolongators,
           smoother,
           smoother_kws,
           nlevels):
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
    prolongators : list
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
    x,_ = smoother(A=A,b=b,x0=x0,
                   **smoother_kws)
    # compute residual
    r = b - A@x
    # compute coarse grid correction
    if lvl == nlevels-1:
        x_c = spsolve( prolongators[lvl].T@r )
    else:
        x_c,_ = vcycle(A=A, b=prolongators[lvl].T@r, x0=None,
                       lvl=lvl+1, prolongators=prolongators,
                       smoother=smoother, smoother_kws=smoother_kws,
                       nlevels=nlevels)
    # interpolate
    x = x + prolongators@x_c
    # post-smooth
    x,info = smoother(A,b,x0=x,**smoother_kws)
    return x,info