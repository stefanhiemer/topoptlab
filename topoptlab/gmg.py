from typing import Any, Callable, Dict, List, Tuple, Union
from itertools import chain
import numpy as np
from scipy.sparse import csc_array




def create_gmg(A: csc_array,
               interpol: Callable,
               nlevels: int) -> Tuple[np.ndarray,int]:
    """
    Create a generic geometric multigrid (GMG) solver for the linear problem
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
    nlevels : int
        number of grid levels.

    Returns
    -------
    prolongators : list of sparse arrays
        hierarchy of prolongators.

    """
    return

def create_interpolator(nelx: int, nely: int,nelz: Union[None,int] = None,
                        ndof: int = 1, stride: int = 2,
                        shape_fncts: Union[None,Callable] = None) -> csc_array:
    # number of dofs
    if nelz is None:
        n = (ndof, nelx+1, nely+1)
    else:
        n = (ndof, nelx+1, nely+1, nelz+1)
    n_dofs = np.prod(n)
    n_nds = np.prod( n[1:] )
    # get shape functions
    if shape_fncts is None and nelz is None:
        from topoptlab.elements.bilinear_quadrilateral import shape_functions
        shape_fncts = shape_functions
    elif shape_fncts is None and nelz is not None:
        from topoptlab.elements.trilinear_hexahedron import shape_functions
        shape_fncts = shape_functions
    # 
    nd_id = np.repeat(np.arange( n_nds ),ndof)
    # find coordinates of each node
    if nelz is None:
        x,y = np.divmod(nd_id,nely+1) 
    else:
        z,rest = np.divmod(nd_id,(nelx+1)*(nely+1))
        x,y = np.divmod(rest,(nely+1))
    # find relative coordinates within each coarse grid cell. 
    # The coarse cell is defined as reaching from -1 to 1 
    x_rel = x%stride * 2/stride - 1
    y_rel = y%stride * 2/stride - 1
    if nelz is not None:
        z_rel = z%stride * 2/stride - 1
    else:
        z_rel=None
    # interpolation weights
    w = shape_fncts(xi=x_rel, eta=y_rel, zeta=z_rel)
    
    return

def create_coarse_inds(nelx: int, nely: int,nelz: Union[None,int] = None,
                       ndof: int = 1, stride: int = 2) -> np.ndarray:
    
    #
    if nelz is None:
        n = (ndof, nelx+1, nely+1)
    else:
        n = (ndof, nelx+1, nely+1, nelz+1)
    #
    idx = np.arange(np.prod(n)).reshape(n,order="F")
    #
    if nelz is None:
        return idx[:,::stride,::stride].flatten(order="F")
    else:
        return idx[:,::stride,::stride,::stride].flatten(order="F")

def create_coarse_mask(nelx: int, nely: int,nelz: Union[None,int] = None,
                       ndof: int = 1, stride: int = 2) -> np.ndarray:
    
    if nelz is None:
        n = np.prod( (ndof,nelx+1,nely+1))
    else:
        n = np.prod( (ndof,nelx+1,nely+1,nelz+1))
    inds = create_coarse_inds(nelx=nelx, nely=nely, nelz = nelz,
                              ndof = ndof, stride = stride)
    mask = np.zeros( n, dtype=bool )
    mask[inds] = True
    return mask