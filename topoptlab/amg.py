from typing import Any, Callable, Dict, List, Tuple, Union
from itertools import chain
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
    A : scipy.sparse.sparse_array
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
                         ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,
                                    np.ndarray,np.ndarray,np.ndarray]:
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
    a_{ij} of matrix A and the negative off-diagonal entries of the i-th row A_{i} 
    of matrix A the following is true:
        
    a_{ij} >= c_pos *max(A_{i}) with a_{ij}>0

    Parameters
    ----------
    A : scipy.sparse.sparse_array
        sparse matrix for which to find coupling of size (nvars,nvars)
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
    s : list of len nvars
        strong couplings for each variable. Each element contains array of 
        indices.
    s_t : list of len nvars
        strong transpose couplings for each variable. Each element contains 
        array of indices.
    iso : np.ndarray
        indices of isolated variables.

    """
    # variables indices
    nvars = A.shape[0]
    var_inds = np.arange(nvars)
    # extract and eliminate diagonal
    diagonal = A.diagonal()
    A.setdiag(0)
    A.eliminate_zeros()
    # off diagonal minimum in each row (this is identical to the largest 
    # negative element by norm)
    min_row = A.min(axis=1).todense()
    if c_pos:
        max_row = A.power(2).sqrt().max(axis=1).todense()
    # extract indices and values
    row,col = A.nonzero()
    val = A[ row,col ]
    # find negative entries for positive/negative coupling
    mask_neg = val < 0
    # convert max/min for element-wise comparison
    row_nr, count = np.unique(row, return_counts=True)
    inds = var_inds[~np.isin(var_inds,row_nr)]
    inds[inds>count.shape[0]] = count.shape[0]
    count = np.insert(arr=count,
                      obj=inds,
                      values=0 )
    min_row = np.repeat(a=min_row, repeats=count)
    max_row = np.repeat(a=max_row, repeats=count)
    # find strong couplings
    mask_strong = np.zeros(mask_neg.shape,dtype=bool)
    mask_strong[mask_neg] = val[mask_neg] <= c_neg*min_row[mask_neg]
    if c_pos:
        mask_strong[~mask_neg] = val[~mask_neg] >= c_pos*max_row[~mask_neg]
    # set of strong couplings
    row_nr, inds = np.unique(row[mask_strong], return_index=True)
    s = np.split(col[mask_strong],inds[1:])
    # insert empty lists for isolated variables that are not strongly coupled 
    # to any other variable
    iso = [0] + var_inds[~np.isin(var_inds,row_nr) ].tolist() + [nvars]
    s = chain.from_iterable([s[i:j]+[[]] for i,j in zip(iso[:-1],iso[1:])])
    s = list(s)[:-1]
    # set of transpose couplings
    inds = np.argsort(col)
    col_nr, split_inds = np.unique(col[inds][mask_strong[inds]], 
                                   return_index=True)
    s_t = np.split(row[inds][mask_strong[inds]], split_inds[1:] )
    # insert empty lists for isolated variables that are not strongly transpose 
    # coupled to any other variable
    iso = [0] + var_inds[~np.isin(var_inds,col_nr) ].tolist() + [nvars]
    s_t = chain.from_iterable([s_t[i:j]+[[]] for i,j in zip(iso[:-1],iso[1:])])
    s_t = list(s_t)[:-1]
    # re-insert diagonal
    A.setdiag(diagonal)
    return row, col, mask_strong, s, s_t, iso[1:-1]

def standard_coarsening(A: csc_array,
                        coupling_fnc: Callable = rubestueben_coupling,
                        coupling_kw: Dict = {"c_neg": 0.2, "c_pos": 0.5},
                        **kwargs: Any):
    """
    Standard coarsening according to page 64 and following in 
    
    Stuebgen, Klaus. "Algebraic multigrid (AMG): an introduction with
    applications." GMD report (1999).
    
    Parameters
    ----------
    A : scipy.sparse.sparse_array
        sparse matrix for which to find coupling of size (nvars,nvars)
    coupling_fnc : callable
        function that determines strong coupling between variables.
    coupling_kw : dictionary
        dictionary containing arguments needed for the coupling function.

    Returns
    -------
    mask_coarse : np.ndarray
        mask for coarse variaables shape (nvars).
    """
    # 
    nvars = A.shape[0]
    # get strong couplings
    row, col, mask_strong, s, s_t, iso = coupling_fnc(A, **coupling_kw)
    #
    mask_coarse = np.zeros(nvars, dtype=bool)
    mask_fine = np.zeros(nvars, dtype=bool)
    undecided = np.ones(nvars, dtype=bool)
    # calculate importance first time (no fine variables here, all variables 
    # are undecided)
    importance = np.zeros(A.shape[0])
    np.add.at( importance, col, mask_strong )
    # convert isolated variables to fine variables
    mask_fine[iso] = True
    # number of undecided variables
    n_u = A.shape[0] - len(iso)
    while n_u > 0:
        # choose variable from the undecided variables with highest importance
        ind = np.argmax( importance )
        # pick strongly coupled variables of new coarse variable that are still 
        # undecided
        _s = s[ind][undecided[s[ind]]]
        # pick set of strong tranpose variables that are still undecided
        _s_t = s_t[ind]
        _s_t = _s_t[undecided[_s_t]]
        # change variable to coarse
        mask_coarse[ind] = True
        # change its transpose coupled variables to fine 
        mask_fine[ _s_t ] = True # possibly improveable
        # take it out of undecided variables
        undecided[ind] = False
        undecided[_s_t] = False
        # reduce importance of other undecided variables due to new coarse 
        # variable
        importance[ _s ] = importance[_s] - 1
        # increase importance of other variables due to new fine variables
        #print(np.hstack( [s[var] for var in _s_t]))
        if len(_s_t) != 0:
            np.add.at (importance, # array to add to
                       np.hstack( [s[var] for var in _s_t]), # indices
                       1.) #value added
        # set importance to zero for new coarse and fine variables
        importance[ind] = 0
        importance[_s_t] = 0
        # update number of undecided variables
        n_u = n_u - 1 - _s_t.shape[0]
    return mask_coarse

def direct_interpolation(A: csc_array, mask_coarse: np.ndarray) -> csc_array:
    """
    Implements page 70 of 
    
    Stuebgen, Klaus. "Algebraic multigrid (AMG): an introduction with
    applications." GMD report (1999).
    
    """
    # extract indices and values
    row,col = A.nonzero()
    val = A[ row,col ]
    # filter out rows of coarse variables
    val = val[~mask_coarse[row]]
    col = col[~mask_coarse[row]]
    row = row[~mask_coarse[row]]
    # negative mask
    mask_neg = val < 0
    # rescaling to "conserve" energy
    denominator, numerator = np.zeros(A.shape[0]), np.zeros(A.shape[0])
    np.add.at(numerator,
              row[mask_neg], 
              val[mask_neg])
    np.add.at(denominator,
              row[mask_neg & mask_coarse[col]], 
              val[mask_neg & mask_coarse[col]])
    # this is alpha on page 70 
    neg_scale = numerator / denominator
    #
    if ( mask_coarse & ~mask_neg ).any():
        # erase previous data
        denominator[:], numerator[:] = 0.
        np.add.at(numerator,
                  row[~mask_neg], 
                  val[~mask_neg])
        np.add.at(denominator,
                  row[~mask_neg & mask_coarse[col]], 
                  val[~mask_neg & mask_coarse[col]])
        pos_scale = numerator / denominator
    #
    prolongator = csc_array()
    return 1#prolongator

def weight_trunctation():
    return

if __name__ == "__main__":
    
    test = np.array([[1., 0., -0.25, -1., 0.55, 0.1, 0.],
                     [0., 1., 0., 0., 0., 0., 0.],
                     [-0.25, 0., 1., 0., 0., 0., 0.],
                     [-1., 0., 0., 2., -1.2, -0.1, 0.],
                     [0.55, 0., 0., -1.2, 5, -2.2, 0.],
                     [0.1, 0., 0., -0.1, -2.2, 1., 0.], 
                     [0., 0., 0., 0., 0., 0., 1.]] )
    #
    test = csc_array(test)
    r,c,mask_strong,s,s_t,iso = rubestueben_coupling(A=test, 
                                                     c_neg = 0.2, 
                                                     c_pos = 0.5)
    print(r,c)
    print(mask_strong)
    print("s",s)
    print("s_t",s_t)
    
    mask_coarse = standard_coarsening(test,
                                      coupling_fnc=rubestueben_coupling,
                                      coupling_kw = {"c_neg": 0.2, 
                                                     "c_pos": 0.5})
    
    print("mask coarse:", mask_coarse)
