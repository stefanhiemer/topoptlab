# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, List, Tuple, Union
from itertools import chain
import numpy as np
from scipy.sparse import csc_array, sparray

def rubestueben_coupling(A: sparray,
                         c_neg: float = 0.2, 
                         c_pos: Union[None,float] = 0.5,
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
        # REFACTOR: this should later be replaced by a more efficient call
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
    s = [np.array(item,dtype=np.int32) for item in s]
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
    s_t = [np.array(item,dtype=np.int32) for item in s_t]
    # re-insert diagonal
    A.setdiag(diagonal)
    return row, col, mask_strong, s, s_t, iso[1:-1]

def standard_coarsening(A: sparray,
                        coupling_fnc: Callable = rubestueben_coupling,
                        coupling_kw: Dict = {"c_neg": 0.2, "c_pos": 0.5},
                        **kwargs: Any) -> np.ndarray:
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
        mask for coarse variables shape (nvars).
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
    importance = np.zeros(A.shape[0],dtype=int)
    np.add.at( importance, col, mask_strong )
    # convert isolated variables to fine variables
    mask_fine[iso] = True
    undecided[iso] = False
    importance[iso] = -1
    # number of undecided variables
    n_u = A.shape[0] - len(iso)
    while n_u > 0:
        # choose variable from the undecided variables with highest importance
        ind = np.argmax( importance )
        # pick strongly coupled variables of new coarse variable that are still 
        # undecided
        _s = s[ind][undecided[s[ind]]]
        # pick set of strong transpose variables that are still undecided
        _s_t = s_t[ind][undecided[s_t[ind]]]
        # change variable to coarse and make it decided
        mask_coarse[ind] = True
        undecided[ind] = False
        importance[ind] = -1
        # reduce importance of other undecided variables due to new coarse 
        # variable
        importance[ _s ] = importance[_s] - 1
        # increase importance of undecided variables due to new fine variables
        #print(np.hstack( [s[var] for var in _s_t]))
        if len(_s_t) != 0:
            # change its transpose coupled variables to fine and take out of 
            # undecided variables
            mask_fine[_s_t] = True
            undecided[_s_t] = False
            importance[_s_t] = -1
            #
            targets = np.hstack([s[var] for var in _s_t])
            # filter undecided only
            targets = targets[undecided[targets]]  
            if targets.size:
                np.add.at(importance, # array to add to
                          np.unique(targets), # indices
                          1) #value added
        # update number of undecided variables
        n_u = n_u - 1 - _s_t.shape[0]
        
    return mask_coarse

def direct_interpolation(A: sparray, mask_coarse: np.ndarray) -> sparray:
    """
    Implements page 70 of 
    
    Stuebgen, Klaus. "Algebraic multigrid (AMG): an introduction with
    applications." GMD report (1999).
    
    Parameters
    ----------
    A : scipy.sparse.sparse_array
        sparse matrix for which to find coupling of size (nvars,nvars)
    mask_coarse : np.ndarray
        has nc True entries and is True for coarse degrees of freedom

    Returns
    -------
    prolongator : scipy.sparse.csc_array
        sparse matrix used to interpolate fine scale degrees of freedom. 
        Interpolation is done by P u_c where P is the prolongator matrix of 
        shape (nvars,nc).
    """
    # convenience 
    mask_fine = ~mask_coarse
    #
    nc = mask_coarse.sum()
    #
    diagonal = A.diagonal()
    # extract indices and values
    row,col = A.nonzero()
    val = A[ row,col ]
    # get off-diagonal
    offdiagonal = row!=col
    row,col,val = row[offdiagonal], col[offdiagonal], val[offdiagonal]
    # filter out rows of coarse variables
    val = val[mask_fine[row]]
    col = col[mask_fine[row]]
    row = row[mask_fine[row]]
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
    # 
    neg_scale = np.zeros(A.shape[0])
    neg_scale[mask_fine] = - numerator[mask_fine] / denominator[mask_fine] \
                           / diagonal[mask_fine]
    #if np.isnan(neg_scale[mask_fine]).any():
    #    print("val: ", val)
    #    print("diagonal: ",diagonal)
    #    print("numerator: ", numerator)
    #    print("denominator: ",denominator)
    #    raise ValueError()
    #
    if ( mask_coarse[col] & ~mask_neg ).any():
        # erase previous data
        denominator[:], numerator[:] = 0.,0.
        np.add.at(numerator,
                  row[~mask_neg], 
                  val[~mask_neg])
        np.add.at(denominator,
                  row[~mask_neg & mask_coarse[col]], 
                  val[~mask_neg & mask_coarse[col]])
        # 
        pos_scale = np.zeros(A.shape[0])
        pos_scale[mask_fine] = -numerator[mask_fine] / denominator[mask_fine] \
                               / diagonal[mask_fine]
    else:
        pos_scale = None
    # filter out columns with fine scale variable
    mask_neg = mask_neg[mask_coarse[col]]
    val = val[mask_coarse[col]]
    row = row[mask_coarse[col]]
    col = col[mask_coarse[col]]
    # rescale 
    val[mask_neg] *= neg_scale[row[mask_neg]]
    if pos_scale is not None:
        val[~mask_neg] *= pos_scale[row[~mask_neg]]
    # re-index the columns as in the columns fine scale variables do not appear
    _,inv = np.unique(col,return_inverse=True)
    col = np.arange(nc)[inv]
    # set diagonal for coarse dofs to one
    row = np.append(row, np.arange(A.shape[0])[mask_coarse])
    col = np.append(col, np.arange(nc))
    val = np.append(val, np.ones(nc))
    return csc_array((val, (row, col)), shape=(A.shape[0],nc))

def create_interpolators_amg(A: sparray,
                             interpol_fnc: Callable = direct_interpolation,
                             interpol_kw: Dict = {},
                             coupling_fnc: Callable = rubestueben_coupling,
                             coupling_kw: [None,Dict] = {"c_neg": 0.2, 
                                                         "c_pos": 0.5},
                             cf_splitting_fnc: Callable = standard_coarsening,
                             cf_splitting_kw: Dict = {},
                             wght_trunc_fnc: Union[None,Callable] = None,
                             wght_trunc_kw: Dict = {},
                             nlevels: int = 2,
                             _lvl: Union[None,int] = None,
                             **kwargs: Any) -> List[sparray]:
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
    interpol_fnc : callable
        interpolation function that with the mask for coarse variables/dofs 
        ultimately constructs the interpolator
    interpol_kw : dict
        keywords for the interpolation function that ultimately constructs the 
        interpolator.
    coupling_fnc : callable
        method to find (strong) couplings which may be used to find coarse 
        variables.
    coupling_kw : callable
        keywords to find (strong) couplings which may be used to find coarse 
        variables.
    cf_splitting_fnc : callable
        function to find (strong) couplings which are used to find coarse 
        variables.
    cf_splitting_kw : dict
        keywords to find (strong) couplings which are used to find coarse 
        variables.
    wght_trunc_fnc : None or callable
        function to truncate weights from the interpolator constructed from
        the interpolation function.
    wght_trunc_kw : None or dict
        keywords to truncate weights from the interpolator constructed from
        the interpolation function.
    nlevels : int
        number of grid levels. Smallest number possible is 2 (one coarse and 
        one fine grid).
    _lvl : None or int
        current level. Do not set when calling this function as it is used to 
        construct the interpolators. 

    Returns
    -------
    interpolators : list of scipy.sparse.csc_array
        hierarchy of interpolator.

    """
    #
    if nlevels < 2:
        raise ValueError("nlevels must be >= 2. nlevels: ",nlevels)
    # 
    if _lvl is None:
        _lvl = 0
    # determine C/F split
    mask_coarse = cf_splitting_fnc(A,
                                   coupling_fnc = coupling_fnc,
                                   coupling_kw = coupling_kw,
                                   **cf_splitting_kw)
    # create interpolatation matrix/array via interpolation function
    interpolator = interpol_fnc(A=A, 
                                mask_coarse=mask_coarse,
                                **interpol_kw)
    # truncate weights
    if wght_trunc_fnc:
        interpolator = wght_trunc_fnc(A=interpolator,**wght_trunc_kw)
    #
    if _lvl == nlevels - 2:
        return [interpolator]
    else:
        interpolators = create_interpolators_amg(
                                     A = interpolator.T@A@interpolator,
                                     interpol_fnc=interpol_fnc,
                                     interpol_kw = interpol_kw,
                                     coupling_fnc = coupling_fnc,
                                     coupling_kw = coupling_kw,
                                     cf_splitting_fnc = cf_splitting_fnc,
                                     cf_splitting_kw = cf_splitting_kw,
                                     wght_trunc_fnc = wght_trunc_fnc,
                                     wght_trunc_kw = wght_trunc_kw,
                                     nlevels = nlevels,
                                     _lvl = _lvl+1)
        return [interpolator] + interpolators
    
def rigid_bodymodes(coords : np.ndarray,
                    pbc : Union[bool,List] = False) -> np.ndarray:
    """
    Calculate rigid body modes from the coordinates of N nodes. 
    
    This function does not take into account any boundary conditions, so modes 
    eliminated by Dirichlet boundary conditions have to be eliminated later. 
    If periodic boundary conditions are active, it is assumed that they are 
    already applied and no redundant nodes are in the list of coordinates.

    Parameters
    ----------
    coords : np.ndarray
        x coordinates shape (N,ndim)
    pbc : bool or list
        bool flags whether dimension is periodic.
        
    Returns
    -------
    B : np.ndarray
        rigid body modes of shape (ndim*N, (ndim**2 + ndim)/2.

    """
    #
    N,ndim = coords.shape
    #
    if isinstance(pbc, bool):
        pbc = tuple([pbc])*ndim
    if np.any(pbc):
        raise NotImplementedError("pbc not yet implemented here.")
    #
    n_modes = int( (ndim**2 + ndim)/2 )
    #
    B = np.zeros((N*ndim,n_modes))
    # translation
    for i in np.arange(ndim):
        B[0+i::ndim,i] = 1.
    # rotation
    if ndim == 3:
        for i in np.arange(n_modes-ndim):
            B[(1+i)%ndim::ndim,i+ndim] = -coords[:,(2+i)%ndim]
            B[(2+i)%ndim::ndim,i+ndim] = coords[:,(1+i)%ndim]
    else:
        B[0::2,2] = -coords[:,1]
        B[1::2,2] = coords[:,0]
    return B
    