# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, List

import numpy as np
from functools import partial
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

def _apply_blockdiagonal(b: np.ndarray, 
                         block_inds: List[np.ndarray],
                         solvers: List[Callable],
                         **kwargs: Any) -> np.ndarray:
    """
    Apply block diagonal preconditioner where the matrix is solved 
    independently on each block via the provided solvers (e. g. direct 
    factorization or even an incomplete one).

    Parameters
    ----------
    b : np.ndarray
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
    x = np.zeros(b.shape)
    for idx, solver in zip(block_inds, solvers):
        x[idx] = solver(b[idx])
    return x

def make_blockdiagonal_preconditioner(A: sparray,
                                      block_inds: List[np.ndarray],
                                      solver_func: Callable,
                                      **kwargs: Any) -> LinearOperator:
    """
    Create a block-diagonal preconditioner where the matrix is split into 
    a number of independent blocks which are all solved independently 
    typically via a factorization or an incomplete one. This is 
    often used as preconditioner for iterative methods like 
    Krylow subspace solvers.

    Parameters
    ----------
    A : sparray
        matrix to be solved.
    block_inds : List[np.ndarray]
        list containing of indices for each block.
    solver_func : Callable
        function to solve/factorize each block.

    Returns
    -------
    blocksparse_preconditioner : scipy.sparse.linalg.LinearOperator
        blocksparse preconditioner.

    """
    # create solver for each block
    solvers = [solver_func(A[idx, :][:, idx]) for idx in block_inds]
    #
    if hasattr(solvers[0], "solve") and callable(getattr(solvers[0], "solve")):
        solvers = [solver.solve for solver in solvers]
    #
    return LinearOperator(A.shape, 
                          matvec=partial(_apply_blockdiagonal, 
                                         block_inds=block_inds, 
                                         solvers=solvers))