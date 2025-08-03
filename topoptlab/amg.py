from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve

def create_amg(A: csc_array,
               interpol: Callable,
               strong_coupling: Callable,
               cf_splitting: Callable,
               weight_trunctation: Union[Callable,None],
               symmetric: bool,
               nlevels: int) -> Tuple[np.ndarray,int]:
    """
    Create a generic algebraic multigrid (AMG) solver for the linear problem
    Ax=b. The key ingredients in are i) coarse/fine splitting ii) interpolation
    method. The first is typically based on some notion of the
    strength/importance of connections between two variables via the matrix
    entry a_{ij} of the matrix A.

    Parameters
    ----------
    A : scipy.sparse.csc_array
        system matrix (e. g. stiffness matrix)
    interpol : callable
        interpolation method.
    strong_coupling : callable
        method to find strong couplings.
    cf_splitting : callable
        method for splitting into fine a coarse variables.
    weight_trunctation : callable or None
        .
    symmetric : bool
        wether A is symmetric.
    nlevels : int
        number of grid levels.

    Returns
    -------
    prolongators : list of sparse arrays
        hierarchy of prolongators.

    """
    # extract and eleminate diagonal
    diagonal = A.diagonal()
    A.setdiag(0)
    A.eliminate_zeros()
    # off diagonal maximum in each row
    max_row = A.power(2).sqrt().max(axis=1).todense()
    # extract indices and values
    i,j = A.nonzero()
    val = A[ i,j ]
    #
    prolongators = []
    # create first prolongator from A

    # create subsequent prolongators by recurrence
    for level in np.arange(nlevels-1):
        # determine C/F split

        # create prolongator via interpolation function

        #
        #prolongators.append(  )
        pass
    return prolongators

def rubestuebgen_coupling(row: np.ndarray, val: np.ndarray, max_row: np.ndarray,
                          c_neg: float = 0.2, c_pos: Union[None,float] = 0.5,
                          **kwargs: Any) -> Union[np.ndarray,np.ndarray]:
    """
    Ruge-Stüben method to determine strong/weak coupling between variables in
    sparse array/matrix A. Works on the row index and the value of an entry of A
    to determine its coupling strength according to Eqs. 115 and  119 in

    Stuben, Klaus. "Algebraic multigrid (AMG): an introduction with
    applications." GMD report (1999).

    Parameters
    ----------
    row : np.ndarray
        row index of entry
    val : np.ndarray
        value of entry.
    max_row : np.ndarray
        maximum for each row.
    c_neg : float
        constant to determine strong coupling for negative variables.
    c_pos : float or None
        constant to determine strong coupling for positive variables.

    Returns
    -------
    mask_strong : np.ndarray
        True if the variable of val is strongly linked to the variable in the
        respective row.
    mask_neg : int
        True if val is negative.

    """
    # find negative entries for positive/negative coupling
    mask_neg = val < 0
    #
    row_nr, count = np.unique(row,return_counts=True)
    max_row = np.repeat(a=max_row, repeats=count)
    # find strong couplings
    mask_strong = np.zeros(mask_neg.shape,dtype=bool)
    mask_strong[mask_neg] = val[mask_neg] <= (-c_neg)*max_row[mask_neg]
    if c_pos:
        mask_strong[~mask_neg] = val[~mask_neg] >= c_pos*max_row[~mask_neg]
    return mask_strong, mask_neg

def direct_interpolation():
    return

def standard_coarsening():

    return

def weight_trunctation():
    return
