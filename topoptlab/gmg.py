# SPDX-License-Identifier: GPL-3.0-or-later
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
                        ndof: int = 1, stride: Union[int,Tuple] = 2, 
                        pbc: Union[Tuple,bool] = False,
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
        coarsening factor in each coordinate direction. Defines the "stride"
        between coarse grid nodes.
    pbc : bool or tuple
        flags for periodic boundary conditions.
    shape_fncts : callable
        Shape function evaluator. If None, bilinear quadrilateral (2D) or
        trilinear hexahedron (3D) shape functions are used.
    
    Returns
    -------
    interpolator : scipy.sparse.sp_array
        Interpolation operator mapping coarse grid values to fine grid values.
    """
    #
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    # convert pbc to tuple
    if isinstance(pbc, bool):
        pbc = (pbc,pbc,pbc)[:ndim]
    # convert stride to tuple
    if isinstance(stride,int):
        stride = (stride,stride,stride)[:ndim]
    # get shape functions
    if shape_fncts is None and nelz is None:
        from topoptlab.elements.bilinear_quadrilateral import shape_functions
        shape_fncts = shape_functions
    elif shape_fncts is None and nelz is not None:
        from topoptlab.elements.trilinear_hexahedron import shape_functions
        shape_fncts = shape_functions
    # 
    offset = [int(not bc) for bc in pbc]
    # number of dofs and coarse dofs
    if nelz is None:
        n = (ndof, 
             nelx+offset[0], 
             nely+offset[1])
        nc = (ndof, 
              int(nelx/stride[0] + offset[0]), 
              int(nely/stride[1] + offset[1])) 
    else:
        n = (ndof, 
             nelx+offset[0], 
             nely+offset[1], 
             nelz+offset[2])
        nc = (ndof, 
              int(nelx/stride[0] + offset[0]), 
              int(nely/stride[1] + offset[1]),
              int(nelz/stride[2] + offset[2])) 
    # total and coarse number of dof
    n_dofs = np.prod(n)
    nc_dofs = np.prod(nc)
    n_nds = np.prod(n[1:])
    nc_nds = np.prod(nc[1:])
    # 
    row = np.repeat(np.arange(n_dofs), 2**ndim)
    # 
    col = ndof*np.arange(nc_nds).reshape(nc[-ndim::],order="F")
    col = np.kron(col, np.ones(stride, dtype=int))
    # delete excessive col entries
    slices = [slice(None,size-o*(s-1)) for s,o,size in \
              zip(stride,offset,col.shape)] 
    col = col[tuple(slices)].flatten(order="F")
    col = np.tile(col[:,None],(1,2**ndim))
    col[:,:3] += ndof*np.array([[1,nc[2]+1,nc[2]]])
    if ndim == 3:
        col[:,4:] += ndof*np.array([ [nc[1]*nc[2]+1, (1+nc[1])*nc[2]+1,
                                      (1+nc[1])*nc[2], nc[1]*nc[2]] ])
    col = (col[:,None,:] + np.arange(ndof)[None,:,None]).flatten("C")
    # calculate relative coordinates for interpolation
    if nelz is None:
        # find coordinates of each node
        x,y = np.divmod(np.arange(n_nds),nely+1)
    else:
        z,rest = np.divmod(np.arange(n_nds),(nelx+1)*(nely+1))
        x,y = np.divmod(rest,(nely+1))
    # find relative coordinates within each coarse grid cell. 
    # The coarse cell is defined as reaching from -1 to 1 
    x_rel = (x%stride[0] * 2/stride[0] - 1)
    y_rel = (y%stride[1] * 2/stride[1] - 1) * (-1)
    if nelz is not None:
        z_rel = z%stride[2] * 2/stride[2] - 1
    else:
        z_rel=None
    # interpolation weights
    val = shape_fncts(xi=x_rel, eta=y_rel, zeta=z_rel)
    val = np.tile(val, (1,ndof)).flatten()
    return csc_array((val, (row, col)), shape=(n_dofs,nc_dofs))

def create_coarse_inds(nelx: int, nely: int, nelz: Union[None,int] = None,
                       ndof: int = 1, stride: int = 2) -> np.ndarray:
    """
    Create degree of freedom indices for coarse degrees of freedom for a 
    geometric multigrid (GMG) solver.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    ndof : int 
        number of nodal degrees of freedom.
    stride : int
        coarsening factor in each coordinate direction.

    Returns
    -------
    indices : np.ndarray
        degree of freedom indices of coarse dofs.

    """
    #
    if nelz is None:
        n = (ndof, nely+1, nelx+1)
        ndim = 2
    else:
        n = (ndof, nely+1, nelx+1, nelz+1)
        ndim = 3
    # convert stride to tuple
    if isinstance(stride,int):
        stride = (stride,stride,stride)[:ndim]
    #
    idx = np.arange(np.prod(n)).reshape(n,order="F")
    #
    if nelz is None:
        return idx[:,::stride[1],::stride[0]].flatten(order="F")
    else:
        return idx[:,::stride[1],::stride[0],::stride[2]].flatten(order="F")

def create_coarse_mask(nelx: int, nely: int, nelz: Union[None,int] = None,
                       ndof: int = 1, stride: int = 2) -> np.ndarray:
    """
    Create a boolean mask identifying the coarse-grid degrees of freedom for a 
    geometric multigrid (GMG) solver.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int
        number of elements in z direction.
    ndof : int 
        number of nodal degrees of freedom.
    stride : int
        coarsening factor in each coordinate direction.

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

def check_stride(nelx: int, nely: int, nelz: Union[None,int],
                 stride: int) -> None:
    """
    .

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nelx : int
        number of elements in y direction.
    nelx : int
        number of elements in z direction.
    stride : int
        coarsening factor in each coordinate direction.

    Returns
    -------
    None

    """
    
    if nelx%stride!=0 | nely%stride!=0 & nelz is None:
        raise ValueError("stride incompatible with lattice dimensions.")
    elif nelx%stride!=0 | nely%stride!=0 | nelz!=0:
        raise ValueError("stride incompatible with lattice dimensions.")
    return

if __name__ == "__main__":
    create_interpolator(nelx=4, 
                        nely=4,
                        nelz=None,
                        ndof=1,
                        pbc=False)