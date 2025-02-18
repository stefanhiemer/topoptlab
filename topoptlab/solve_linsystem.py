from itertools import product

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve,cg, spilu, LinearOperator, factorized

from cvxopt import spmatrix,matrix
from cvxopt.cholmod import linsolve

from topoptlab.elements.bilinear_quadrilateral import shape_functions

def solve_lin(K,rhs,solver,
              preconditioner=None,
              P=None):
    """
    Solve linear system Ku=rhs for a generic matrix K with preconditioner P. 
    rhs might contain multiple sets of boundary conditions that are solved 
    sequentially.

    Parameters
    ----------
    K : scipy.sparse.csc_matrix or cvxopt.base.spmatrix (ndof_free,ndof_free)
        element degree of freedom matrix.
    rhs : np.ndarray or cvxopt.base.matrix (ndof_free,ndof_free,nbc)
        rows to delete .

    Returns
    -------
    iM : np.ndarray shape (N)
        row indices for matrix construction.
    jM : np.ndarray shape (N)
        column indices for matrix construction.
    """ 
    # direct solvers
    if solver == "scipy-direct":
        if rhs.shape[1] == 1:
            return spsolve(K, rhs)[:,None], None, None
        else:
            return spsolve(K, rhs), None, None
    # lu decomposition with either with SuperLU or umfpack
    elif solver == "scipy-lu":
        lu_solve = factorized(K)
        if rhs.shape[1] == 1:
            sol =  lu_solve(rhs)[:,None]
        else:
            sol = np.zeros(rhs.shape)
            for i in np.arange(rhs.shape[1]):
                sol[:,i] = lu_solve(rhs)
        return sol, lu_solve, None
    # sparse cholesky decomposition
    elif solver == "cvxopt-cholmod":
        B = matrix(rhs)
        linsolve(K,B)
        return np.array(B), None, None
    
    # iterative solvers
    if P is None and preconditioner is not None:
        if preconditioner == "scipy-ilu":
            ilu = spilu(K, fill_factor=100., drop_tol=1e-5)
            P = LinearOperator(shape=K.shape, 
                               matvec=ilu.solve)
    # without preconditioner, bad idea. Purely there for testing
    if solver == "scipy-cg":
        # more than one set of boundary conditions to solve
        if rhs.shape[1] == 1:
            sol,fail = cg(K, rhs, rtol=1e-10, maxiter=10000,
                          M=P)
            # shape consistent
            sol = sol[:,None]
            if fail != 0:
                raise RuntimeError("cg iteration did not converge.")   
        else:
            sol = np.zeros(rhs.shape)
            for i in np.arange(rhs.shape[1]):
                sol[:,i],fail = cg(K, rhs[:,i], rtol=1e-10, maxiter=10000,
                                   M=P)
                if fail != 0:
                    raise RuntimeError("cg iteration did not converge for bc ",
                                       i) 
                
        return sol, None, P
    else:
        raise ValueError("Unknown solver: ",solver)
