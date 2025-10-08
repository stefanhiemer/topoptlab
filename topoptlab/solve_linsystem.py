# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Dict,Tuple,Union

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve,cg, spilu, LinearOperator, factorized, LaplacianNd

from cvxopt import matrix,spmatrix
from cvxopt.cholmod import solve,symbolic,numeric
from pyamg.aggregation import adaptive_sa_solver,rootnode_solver,smoothed_aggregation_solver,pairwise_solver
from pyamg.classical import air_solver,ruge_stuben_solver

from topoptlab.log_utils import EmptyLogger,SimpleLogger
from topoptlab.linear_solvers import cg as topopt_cg
from topoptlab.linear_solvers import pcg as pcg

def solve_lin(K: Union[csc_array,spmatrix], rhs: Union[np.ndarray,matrix],
              solver: str,
              rhs0: Union[None,np.ndarray,matrix] = None,
              solver_kw: Dict = {},
              preconditioner: Union[None,str] = None,
              P: Union[None,Callable,spmatrix,csc_array] = None,
              preconditioner_kw: Dict = {},
              logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger,
              **kwargs: Any
              ) -> Tuple[np.ndarray, 
                         Union[None,Callable],
                         Union[None,Callable,spmatrix,csc_array]]:
    """
    Solve linear system Ku=rhs for a generic matrix K with preconditioner P.
    rhs might contain multiple sets of boundary conditions that are solved
    sequentially.

    Parameters
    ----------
    K : scipy.sparse.csc_arrayor cvxopt.base.spmatrix (ndof_free,ndof_free)
        matrix to solve.
    rhs : np.ndarray or cvxopt.base.matrix (ndof_free,nbc)
        right hand side of linear system.
    solver : str
        string that indicates the library and type of solver to be used.
    rhs0 : None or np.ndarray or cvxopt.base.matrix (ndof_free,nbc)
        initial guess for right hand side of linear system. Only relevant for
        iterative solvers.
    solver_kw : dict
        arguments for the solver.
    preconditioner : str
        string that indicates preconditioner to be used.
    preconditioner_kw : dict
        arguments for the preconditioner.
    P : callable or sparse matrix format
        preconditioner created during previous solution of Ku. Concrete nature
        depends on the solver and library used.
    logger : EmptyLogger or SimpleLogger
        logger for writing information to logfile.

    Returns
    -------
    solution : np.ndarray shape (ndof_free)
        solution of linear system.
    factorization : callable
        callable that allows to re-apply the factorization of matrix K to
        another right hand side.
    preconditioner : callable or scipy.sparse.matrix or similar object
        preconditioner created during the solution. Concrete nature depends on
        the solver and library used.
    """
    ### direct solvers
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
        factorization = symbolic(K, uplo='L')
        numeric(K, factorization)
        solve(factorization, B)
        #linsolve(K,B)
        return np.array(B), factorization, None
    
    ### iterative solvers
    if P is None and preconditioner is not None:
        if preconditioner == "scipy-ilu":
            ilu = spilu(K, **preconditioner_kw)
            P = LinearOperator(shape=K.shape,
                               matvec=ilu.solve)
        elif preconditioner == "pyamg-air":
            P = air_solver(A=K,
                           **preconditioner_kw).aspreconditioner(cycle='V')
        elif preconditioner == "pyamg-ruge_stuben":
            P = ruge_stuben_solver(A=K,
                                   **preconditioner_kw).aspreconditioner(cycle='V')
        elif preconditioner == "pyamg-smoothed_aggregation":
            P = smoothed_aggregation_solver(A=K,
                                            **preconditioner_kw).aspreconditioner(cycle='V')
        elif preconditioner == "pyamg-rootnode":
            P = rootnode_solver(A=K,
                                **preconditioner_kw).aspreconditioner(cycle='V')
        elif preconditioner == "pyamg-pairwise":
            P = pairwise_solver(A=K,
                                **preconditioner_kw).aspreconditioner(cycle='V')
        elif preconditioner == "pyamg-adaptive_sa":
            P,work = adaptive_sa_solver(A=K,
                                        **preconditioner_kw)
            P = P.aspreconditioner(cycle='V')
    
    # without preconditioner, bad idea. Purely there for testing
    if solver == "scipy-cg":
        # more than one set of boundary conditions to solve
        if rhs.shape[1] == 1:
            sol,fail = cg(A=K, b=rhs,  
                          x0=rhs0,
                          M=P,
                          **solver_kw)
            # shape consistent
            sol = sol[:,None]
            if fail != 0:
                raise RuntimeError("cg iteration did not converge.")
        else:
            sol = np.zeros(rhs.shape)
            for i in np.arange(rhs.shape[1]):
                sol[:,i],fail = cg(A=K, b=rhs[:,i],
                                   x0=rhs0[:,i],
                                   M=P, 
                                   **solver_kw)
                if fail != 0:
                    raise RuntimeError("cg iteration did not converge for bc ",
                                       i)
        return sol, None, P
    elif solver == "topoptlab-pcg":
        # more than one set of boundary conditions to solve
        sol = np.zeros(rhs.shape)
        for i in np.arange(rhs.shape[1]):
            sol[:,i],fail = pcg(A=K, b=rhs[:,i], 
                                x0=rhs0[:,i],
                                P=P, logger=logger,
                                **solver_kw)
            if fail != 0:
                raise RuntimeError("pcg iteration did not converge for bc ",
                                   i)
        return sol, None, P
    elif solver == "topoptlab-cg":
        sol = np.zeros(rhs.shape)
        for i in np.arange(rhs.shape[1]):
            sol[:,i],fail = topopt_cg(A=K, b=rhs[:,i], 
                                      x0=rhs0[:,i],
                                      logger=logger,
                                      **solver_kw)
            if fail != 0:
                raise RuntimeError("cg iteration did not converge for bc ",
                                   i)
        return sol, None, P
    else:
        raise ValueError("Unknown solver: ",solver)

def laplacian(grid: Tuple) -> Tuple[csc_array,np.ndarray]:
    """
    Construct Laplacian on a uniform rectangular grid in N dimensions and the right hand side
    to the linear problem
    
    Lx=b
    
    where the first entry of b is 1 and the last is -1.
    
    
    This is purely intended as a short test problem to check whether solvers actually work.
    
    Parameters
    ----------
    grid : tuple
        number of grid points in each direction. len(grid) is n.

    Returns
    -------
    L : csc_array
        Laplacian
    b : callable
        callable that allows to re-apply the factorization of matrix K to
        another right hand side.
    """
    #
    L = LaplacianNd(grid_shape=grid,
                    boundary_conditions = "neumann").tosparse().tocsc()
    #
    b = np.zeros(L.shape[0]-1)
    b[1] = -1.
    b[-1] = 1.
    #
    L = L[1:,:][:,1:]
    return L.astype(np.float64) * (-1), b