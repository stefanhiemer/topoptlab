from typing import Any, Callable, Dict, List, Tuple, Union
from itertools import chain
import numpy as np
from scipy.sparse import sparray, csc_array

def create_gmg(A: sparray,
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
                        shape_fncts: Union[None,Callable] = None) -> sparray:
    """
    Construct the interpolation (prolongation) operator for geometric
    multigrid (GMG).
    
    The interpolation maps values from coarse grid nodes to fine grid nodes
    using shape functions. The stride determines the spacing between coarse
    grid nodes in each coordinate direction. For example, a stride of 2 means
    that every second fine grid node is designated as a coarse grid node.
    
    Parameters
    ----------
    nelx : int
        Number of elements in the x direction.
    nely : int
        Number of elements in the y direction.
    nelz : int
        Number of elements in the z direction (for 3D problems).
    ndof : int
        Number of degrees of freedom per node. 
    stride : int
        Coarsening factor in each coordinate direction. Defines the "stride"
        between coarse grid nodes.
    shape_fncts : callable
        Shape function evaluator. If None, bilinear quadrilateral (2D) or
        trilinear hexahedron (3D) shape functions are used.
    
    Returns
    -------
    interpolator : scipy.sparse.sp_array
        Interpolation operator mapping coarse grid values to fine grid values.
    """

    # number of dofs
    if nelz is None:
        n = (ndof, nelx+1, nely+1)
    else:
        n = (ndof, nelx+1, nely+1, nelz+1)
    n_dofs = np.prod(n)
    n_nds = np.prod(n[1:])
    # get shape functions
    if shape_fncts is None and nelz is None:
        from topoptlab.elements.bilinear_quadrilateral import shape_functions
        shape_fncts = shape_functions
    elif shape_fncts is None and nelz is not None:
        from topoptlab.elements.trilinear_hexahedron import shape_functions
        shape_fncts = shape_functions
    # 
    nd_id = np.repeat(np.arange( n_nds ),ndof)
    if nelz is None:
        # find coordinates of each node
        x,y = np.divmod(nd_id,nely+1)
        # find column indices
        n2 = ((y-y%stride) + (x-x%stride)*(nely+1))*ndof + np.arange(n_dofs)%ndof
        n1 = n2 + ndof*(nely+1)*stride
        
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

def create_coarse_inds(nelx: int, nely: int, nelz: Union[None,int] = None,
                       ndof: int = 1, stride: int = 2) -> np.ndarray:
    """
    Create degree of freedom indices for coarse degrees of freedom for a 
    geometric multigrid (GMG) solver.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nelx : int
        number of elements in y direction.
    nelx : int
        number of elements in z direction.
    ndof : int 
        number of nodal degrees of freedom.
    stride : int
        Coarsening factor in each coordinate direction.

    Returns
    -------
    indices : sparse arrays
        degree of freedom indices of coarse dofs.

    """
    
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
    """
    Create a boolean mask identifying the coarse-grid degrees of freedom for a 
    geometric multigrid (GMG) solver.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nelx : int
        number of elements in y direction.
    nelx : int
        number of elements in z direction.
    ndof : int 
        number of nodal degrees of freedom.
    stride : int
        Coarsening factor in each coordinate direction.

    Returns
    -------
    mask : np.ndarray
        mask for degree of freedom indices of coarse dofs.

    """
    if nelz is None:
        n = np.prod( (ndof,nelx+1,nely+1))
    else:
        n = np.prod( (ndof,nelx+1,nely+1,nelz+1))
    inds = create_coarse_inds(nelx=nelx, nely=nely, nelz = nelz,
                              ndof = ndof, stride = stride)
    mask = np.zeros( n, dtype=bool )
    mask[inds] = True
    return mask