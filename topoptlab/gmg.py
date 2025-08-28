from typing import Any, Callable, Dict, List, Tuple, Union
from itertools import chain
import numpy as np
from scipy.sparse import csc_array

def create_gmg(A: csc_array,
               interpol: Callable,
               strong_coupling: Callable,
               cf_splitting: Callable,
               weight_trunctation: Union[Callable,None],
               symmetric: bool,
               nlevels: int) -> Tuple[np.ndarray,int]:
    """
    Create a generic geometric multigrid (AMG) solver for the linear problem
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
    return

def create_interpolator(nelx: int, nely: int,nelz: Union[None,int] = None,
                        ndof: int = 1, nlevel: int = 2) -> csc_array:
    #
    if nelz is None:
        n = np.prod((ndof, nelx+1, nely+1))
    else:
        n = np.prod((ndof, nelx+1, nely+1, nelz+1))
    # node_ids
    nd_id = np.arange(n)
    # find coordinates of each element/density
    if nelz is None:
        x,y = np.divmod(nd_id,nely+2) # same as np.floor(el/nely),el%nely
        return x,y
    else:
        z,rest = np.divmod(el,(nelx+1)*(nely+1))
        x,y = np.divmod(rest,(nely+1))
        return x,y,z
    return

def create_coarse_inds(nelx: int, nely: int,nelz: Union[None,int] = None,
                       ndof: int = 1, nlevel: int = 2) -> np.ndarray:
    
    #
    if nelz is None:
        n = (ndof, nelx+1, nely+1)
    else:
        n = (ndof, nelx+1, nely+1, nelz+1)
    #
    idx = np.arange(np.prod(n)).reshape(n,order="F")
    #
    if nelz is None:
        return idx[:,::nlevel,::nlevel].flatten(order="F")
    else:
        return idx[:,::nlevel,::nlevel,::nlevel].flatten(order="F")

def create_coarse_mask(nelx: int, nely: int,nelz: Union[None,int] = None,
                       ndof: int = 1, nlevel: int = 2) -> np.ndarray:
    
    if nelz is None:
        n = np.prod( (ndof,nelx+1,nely+1))
    else:
        n = np.prod( (ndof,nelx+1,nely+1,nelz+1))
    inds = create_coarse_inds(nelx=nelx, nely=nely, nelz = nelz,
                              ndof = ndof, nlevel = nlevel)
    mask = np.zeros( n, dtype=bool )
    mask[inds] = True
    return mask