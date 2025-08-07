from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from scipy.sparse import csc_array

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

def rubestueben_coupling(A: csc_array,
                         c_neg: float = 0.2, c_pos: Union[None,float] = 0.5,
                         **kwargs: Any
                         ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Ruge-Stüben method to determine strong/weak coupling between variables in
    sparse array/matrix A. Works on the row index and the value of an entry of A
    to determine its coupling strength according to Eqs. 115 and  119 in

    Stuebgen, Klaus. "Algebraic multigrid (AMG): an introduction with
    applications." GMD report (1999).
    
    We have slightly modified the expressions (their meaning stays the same): 
    variable i is strongly negatively coupled to variable j if for the entry
    a_{ij} of matrix A and the the off-diagonal entries of the i-th row A_{i} 
    of matrix A the following is true:
    
    a_{ij} <= (-c_neg) *max(A_{i}) with a_{ij}<0
    
    variable i is strongly negatively coupled to variable j if for the entry
    a_{ij} of matrix A and the the off-diagonal entries of the i-th row A_{i} 
    of matrix A the following is true:
        
    a_{ij} >= c_pos *max(A_{i}) with a_{ij}>0

    Parameters
    ----------
    A : np.ndarray
        row index of entry
    c_neg : float
        constant to determine strong coupling for negative variables.
    c_pos : float or None
        constant to determine strong coupling for positive variables.

    Returns
    -------
    row : np.ndarray
        row index
    col : np.ndarray
        column_index
    mask_strong : np.ndarray
        True if the variable of val is strongly linked to the variable in the
        respective row.

    """
    # extract and eliminate diagonal
    diagonal = A.diagonal()
    A.setdiag(0)
    A.eliminate_zeros()
    # off diagonal maximum in each row
    max_row = A.power(2).sqrt().max(axis=1).todense()
    # extract indices and values
    row,col = A.nonzero()
    val = A[ row,col ]
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
    # 
    A.setdiag(diagonal)
    return row, col, mask_strong

def standard_coarsening(A,
                        coupling_fnc=rubestueben_coupling,
                        coupling_kw: Dict = {"c_neg": 0.2, "c_pos": 0.5},
                        seed=0):
    """
    
    
    """
    # get strong couplings
    row, col, mask_strong = coupling_fnc(A, **coupling_kw)
    #
    mask_coarse = np.zeros(mask_strong.shape[0], dtype=bool)
    undecided = np.ones(mask_strong.shape[0], dtype=bool)
    # calculate importance first time (no fine variables here, all variables 
    # are undecided)
    importance = np.zeros(A.shape[0])
    np.add.at( a=importance, indices = col, b=mask_strong )
    # as starting point choose random variable from the variables with highest 
    # importance
    np.random.seed(seed)
    ind = np.random.randint(0,A.shape[0])
    mask_coarse[ind] = True
    undecided[ind] = False
    # number of undecided variables
    n_u = A.shape[0] -1
    while n_u > 0:
        pass
    return mask_coarse

def direct_interpolation():
    return



def weight_trunctation():
    return
