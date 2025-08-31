from typing import Any, Callable, List

import numpy as np
from functools import partial
from scipy.sparse import csc_array
from scipy.sparse.linalg import LinearOperator

def _apply_blocks(x: np.ndarray, 
                  block_inds: List[np.ndarray], 
                  solvers: List[Callable],
                  **kwargs: Any) -> np.ndarray:
    """
    Apply block preconditioner where the matrix is solved independently on each 
    block via the provided solvers (e. g. direct factorization or even an 
    incomplete one).

    Parameters
    ----------
    x : np.ndarray
        initial guess for solution
    block_inds : list
        list containing of indices for each block.
    solvers : list
        list containing solvers to solve each block.

    Returns
    -------
    x_new : np.ndarray
        updated solution.

    """
    #
    for idx, solve in zip(block_inds, solvers):
        x[idx] = solve(x[idx])
    return x

def make_block_preconditioner(A: csc_array,
                              block_inds: List[np.ndarray],
                              solver_func: Callable,
                              symmetric: bool = True,
                              **kwargs: Any) -> LinearOperator:
    """
    Create a block preconditioner where the matrix is split into a number of 
    independent blocks which are all solved independently typically via a 
    direct factorization or even an incomplete one. This is often used as 
    preconditioner for iterative solver like conjugate gradient solvers

    Parameters
    ----------
    A : csc_array
        matrix to be solved.
    block_inds : List[np.ndarray]
        list containing of indices for each block.
    solver_func : Callable
        function to solve/factorize each block.
    symmetric : bool
        assume A to be symmetric

    Returns
    -------
    blocksparse_preconditioner : scipy.sparse.linalg.LinearOperator
        blocksparse preconditioner.

    """
    # factorize each block
    solvers = []
    for k, idx in enumerate(block_inds):
        solvers.append(solver_func(A[idx, :][:, idx], idx, k))
    # 
    matvec = partial(_apply_blocks, 
                     block_inds=block_inds, solvers=solvers)
    #
    return LinearOperator(A.shape, 
                          matvec=matvec, 
                          rmatvec=matvec if symmetric else None, dtype=A.dtype)
