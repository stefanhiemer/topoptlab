# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, List

import numpy as np
from functools import partial
from scipy.sparse import sparray
from scipy.sparse.linalg import LinearOperator

def _apply_blocks(b: np.ndarray, 
                  block_inds: List[np.ndarray], 
                  solvers: List[Callable],
                  **kwargs: Any) -> np.ndarray:
    """
    Apply block preconditioner where the matrix is solved independently on each 
    block via the provided solvers (e. g. direct factorization or even an 
    incomplete one).

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

def make_block_preconditioner(A: sparray,
                              block_inds: List[np.ndarray],
                              solver_func: Callable,
                              **kwargs: Any) -> LinearOperator:
    """
    Create a block preconditioner where the matrix is split into a number of 
    independent blocks which are all solved independently typically via a 
    direct factorization or even an incomplete one. This is often used as 
    preconditioner for iterative solver like conjugate gradient solvers.

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
    solvers = []
    for k, idx in enumerate(block_inds):
        solvers.append(solver_func(A[idx, :][:, idx]))
    #
    if hasattr(solvers[0], "solve") and callable(getattr(solvers[0], "solve")):
        solvers = [solver.solve for solver in solvers]
    #
    return LinearOperator(A.shape, 
                          matvec=partial(_apply_blocks, 
                                         block_inds=block_inds, 
                                         solvers=solvers))

def create_primitive_blocks(A: sparray, 
                            nblocks: int,
                            **kwargs) -> List[np.ndarray]:
    """
    Create block indices by just splitting the dofs in nblocks even-sized 
    chunks.

    Parameters
    ----------
    A : sparray
        matrix to be solved.
    nblocks : int
        function to solve/factorize each block.

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    return np.split(np.arange(A.shape[0]), 
                    np.arange(0, A.shape[0],
                              np.ceil(A.shape[0]/nblocks).astype(int))[1:])