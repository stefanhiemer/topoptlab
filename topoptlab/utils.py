# SPDX-License-Identifier: GPL-3.0-or-later
from warnings import warn
from typing import Any,Dict,List,Tuple,Union

import numpy as np
from scipy.ndimage import zoom
from scipy.linalg import solve_triangular

from topoptlab.optimizer.mma_utils import mma_defaultkws, gcmma_defaultkws

def default_outputkw() -> Dict:
    """
    Return the default dictionary containing general parameters for output
    and logging in the topology_optimization.main() function.

    This dictionary defines the default for output-related behavior, such as 
    file naming, verbosity, etc. It can be used as a reference or as a source 
    for inserting missing keys in user-defined dictionaries.

    Keys and Default Values 
    -----------------------

    +------------------+-------+-----------+---------------------------------------------+
    | Key              | Type  | Default   | Description                                 |
    +------------------+-------+-----------+---------------------------------------------+
    | "file"           | str   | "topopt"  | Base name used for output files.            |
    | "display"        | bool  | True      | Display intermediary design.                |
    | "export"         | bool  | True      | Export final results to disk as VTK.        |
    | "write_log"      | bool  | True      | Write a log file.                           |
    | "profile"        | bool  | False     | Profile code execution.                     |
    | "output_movie"   | bool  | False     | Generate a movie of intermediate designs.   |
    | "verbosity"      | int   | 20        | Level of verbosity (see SimpleLogger).      |
    +------------------+-------+-----------+---------------------------------------------+


    Returns
    -------
    output_kw
        dictionary containing the default output parameters.
    """
    
    return {"file": "topopt",
            "display": True,
            "export": True,
            "write_log": True,
            "profile": False,
            "output_movie": False,
            "verbosity": 20}

def check_output_kw(output_kw: Dict) -> None:
    """
    Check that general output parameters are sensible or implemented and insert
    missing ones based on the default dictionary.

    Parameters
    ----------
    output_kw : dictionary
        contains general parameters for outputting information . All relevant 
        values can be found in default_outputkw.

    Returns
    -------
    None
    
    """
    #
    default_kw = default_outputkw()
    #
    missing_keys = set(default_kw.keys()) - set(output_kw.keys())
    for key in missing_keys:
        output_kw[key] = default_kw[key]
    return

def check_optimizer_kw(optimizer: str, 
                       n : int,
                       ft : int,
                       n_constr : int,
                       optimizer_kw: Dict) -> None:
    """
    Check that general optimuer parameters are sensible or implemented and 
    insert missing ones based on the default dictionary.

    Parameters
    ----------
    optimizer : str
        name of optimizer.
    n : int
        number of design variables.
    ft : int
        code for filter.
    n_constr : int 
        number of constraints.
    optimizer_kw : dictionary
        contains parameters for the optimizern . All relevant 
        values can be found in default_outputkw.

    Returns
    -------
    None
    
    """
    #
    if optimizer_kw is None:
        optimizer_kw = dict()
    #
    if optimizer == "mma":
        default_kw = mma_defaultkws(n = n,
                                    ft = ft,
                                    n_constr = n_constr)
    elif optimizer == "gcmma":
        default_kw = gcmma_defaultkws(n = n,
                                      ft = ft,
                                      n_constr = n_constr)
    elif optimizer in ["oc","ocm","oc88","ocg"]:
        default_kw = {}
    #
    missing_keys = set(default_kw.keys()) - set(optimizer_kw.keys())
    for key in missing_keys:
        optimizer_kw[key] = default_kw[key] 
    return optimizer_kw

def check_simulation_params(simulation_kw: Dict) -> None:
    """
    Check that general simulation parameters are sensible or implemented.

    Parameters
    ----------
    simulation_kw : dictionary
        contains general information about simulation. At the moment, only
        "grid","element order", "meshfile" are supported with contents 
        "regular",1,None.

    Returns
    -------
    None
    
    """
    admissible = [["regular"],
                  [1],
                  [None]]
    keys = ["grid","element order", "meshfile"]
    for i,key in enumerate(keys):
        if not simulation_kw[key] in admissible[i]:
            raise ValueError(f"{key} must be one of those: {admissible[i]}")
    return


def even_spaced_ternary(npoints: int) -> np.ndarray:
    """
    Create even spaced points for a ternary phase system.

    Parameters
    ----------
    npoints : int
        number of points.

    Returns
    -------
    fracs : np.ndarray shape (k+1,3)
        phase fractions.
    """
    
    fracs = [] 
    for i,a in enumerate(np.linspace(0.0,1.0,npoints)):
        for b in np.linspace(0.0,1.0,npoints)[:(npoints-i)]:
            fracs.append([a,b,1-a-b])
    return np.array(fracs)

def unique_sort(iM: np.ndarray, jM: np.ndarray, 
                combine: bool = False) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray]:
    """
    Sort first according to iM, then sort values of equal value iM according 
    to jM.

    Parameters
    ----------
    iM : np.ndarray, shape (n)
        first array.
    jM : np.ndarray, shape (n)
        second array.
    combine : scalar bool
        if True, stack both to to a column array of shape (n,2) 

    Returns
    -------
    iM : np.ndarray shape (n)
        if not combine returns sorted iM
    jM : np.ndarray shape (n)
        if not combine returns sorted jM
    M : np.ndarray shape (n,2)
        if combine returns column stack of sort iM and jM.

    """
    
    inds = np.lexsort((jM,iM))
    if combine:
        return np.column_stack((iM[inds],jM[inds])) 
    else:
        return iM[inds],jM[inds]

def dict_without(dictionary: Dict, keys: List) -> Dict:
    """
    Return a copy of a dictionary without the given keys.
    
    Parameters
    ----------
    dictionary : dict
        full dictionary.
    keys : list
        list of keys to not include in the copy

    Returns
    -------
    dictionary_new : dict
        dictionary without given keys 
    """
    return {k: v for k, v in dictionary.items() if k not in keys}

def map_eltoimg(quant: np.ndarray, 
                nelx: int, nely: int, 
                **kwargs: Any) -> np.ndarray:
    """
    Map quantity located on elements on the usual regular grid to an image.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
        
    Returns
    -------
    img : np.ndarray, shape (nely,nelx,nchannel)
        quantity mapped to image.

    """
    #
    shape = (nely,nelx)+quant.shape[1:]
    return quant.reshape(shape,order="F")

def map_imgtoel(img: np.ndarray, 
                nelx: int, nely: int, 
                **kwargs: Any) -> np.ndarray:
    """
    Map image of quantity back to 1D np.ndarray with correct (!) ordering.

    Parameters
    ----------
    img : np.ndarray, shape (nely,nelx)
        image of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.

    Returns
    -------
    quant : np.ndarray, shape (n)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely])+img.shape[2:]
    return img.reshape(shape,order="F")

def map_eltovoxel(quant: np.ndarray, 
                  nelx: int, nely: int, nelz: int,
                  **kwargs: Any) -> np.ndarray:
    """
    Map quantity located on elements on the usual regular grid to a voxels.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    voxel : np.ndarray shape (nelz,nely,nelx,nchannel)
        quantity mapped to voxels.

    """
    #
    shape = (nelz,nelx,nely)+quant.shape[1:]
    return quant.reshape(shape).transpose((0,2,1)+tuple(range(3,len(shape))))

def map_voxeltoel(voxel: np.ndarray, 
                  nelx: int, nely: int, nelz: int,
                  **kwargs: Any) -> np.ndarray:
    """
    Map voxels of quantity back to on elements on the usual regular grid.

    Parameters
    ----------
    voxel : np.ndarray, shape (nelz,nely,nelx,nchannel)
        voxels of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    quant : np.ndarray, shape (n,nchannel)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely*nelz])+voxel.shape[3:]
    voxel = voxel.transpose((0,2,1)+tuple(range(3,len(voxel.shape))))
    return voxel.reshape(shape)

def elid_to_coords(el: np.ndarray, 
                   nelx: int, nely: int, nelz: Union[None,int] = None,
                   l: Union[float,List] = [1.,1.,1.],
                   **kwargs: Any) -> np.ndarray:
    """
    Map element IDs to cartesian coordinates in the usual regular grid.

    Parameters
    ----------
    el : np.ndarray, shape (n)
        element IDs.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    l : float or list 
        side length of element
    
    Returns
    -------
    coords : np.ndarray 
        element coordinates of shape shape (n,ndim).

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if isinstance(l, float):
        l = [l for i in range(ndim)]
    # find coordinates of each element/density
    if ndim == 2:
        x,y = np.divmod(el,nely) # same as np.floor(el/nely),el%nely
        coords = (np.array([[0,nely-1]])-np.column_stack((x,y)))
        #coords = coords * np.array([[-1,1]]) * np.array(l)[None,:ndim] 
    else:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
        coords = (np.array([[0,nely-1,0]])-np.column_stack((x,y,z)))
        #coords = coords * np.array([[-1,1,1]]) * np.array(l)[None,:] 
    return coords * np.array([[-1,1,-1]])[:,:ndim] * np.array(l)[None,:ndim] 
    
def nodeid_to_coords(nd: np.ndarray, 
                     nelx: int, nely: int, nelz: Union[None,int] = None,
                     l: Union[float,List] = [1.,1.,1.],
                     **kwargs: Any) -> np.ndarray:
    """
    Map node IDs to cartesian coordinates in the usual regular grid.

    Parameters
    ----------
    nd : np.ndarray, shape (n)
        node IDs.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.
    l : float or List
        side length of element
        
    Returns
    -------
    coords : np.ndarray 
        node coordinates of shape shape (n,ndim).

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if isinstance(l, float):
        l = [l for i in range(ndim)]
    # find coordinates of each element/density
    if nelz is None:
        x,y = np.divmod(nd,nely+1) # same as np.floor(el/nely),el%nely
        coords = np.array([[1,nely]])-np.column_stack((x,y))-0.5
        #coords = coords * np.array([[-1,1]]) * np.array(l)[None,:]
    else:
        z,rest = np.divmod(nd,(nelx+1)*(nely+1))
        x,y = np.divmod(rest,(nely+1))
        coords = np.array([[1,nely,1]])-np.column_stack((x,y,z))-0.5
        #coords = coords * np.array([[-1,1,1]]) * np.array(l)[None,:]
    return coords * np.array([[-1,1,-1]])[:,:ndim] * np.array(l)[None,:ndim]

def upsampling(x: np.ndarray, magnification: Union[float,int,List],
               nelx: int, nely: int, nelz: Union[None,int] = None,
               return_flat: bool = True, order: int = 0) -> np.ndarray:
    """
    Upsample current design variables defined on the standard regular grid to 
    a larger design by interpolation. With order 0 the design is replicated on
    a finer scale in a volume conserving fashion, otherwise spline 
    interpolation might violate this.

    Parameters
    ----------
    x : np.ndarray shape (n,nchannel)
        design variables.
    magnification : float
        magnification factor.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None, optional
        number of elements in z direction. The default is None.
    return_flat : bool, optional
        return the design variables flattened. If false returns an image or a 
        voxel graphic. The default is True.
    order : int, optional
        order of spline interpolation for upsampling. The default is 0.

    Returns
    -------
    x_new : np.ndarray shape (n) or shape (nely,nelx) or shape (nelz,nely,nelx)
        upsampled design variables.

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    if isinstance(magnification, (float,int)):
        magnification = ndim * [magnification] + [1]
    #
    if nelz is None:
        x = map_eltoimg(quant=x, 
                        nelx=nelx, nely=nely)
    else:
        x = map_eltovoxel(quant=x, 
                          nelx=nelx, nely=nely, nelz=nelz)
    #
    x = zoom(x, zoom=magnification,
             order=order, mode="nearest",
             cval=0.)
    #
    if return_flat:
        if nelz is None:
            nely,nelx,nchannel = x.shape
            x = map_imgtoel(img=x, 
                            nelx=nelx, nely=nely)
        else:
            nelz,nely,nelx,nchannel = x.shape
            x = map_voxeltoel(voxel=x, 
                              nelx=nelx, nely=nely, nelz=nelz)
    return x

def svd_inverse(A: np.ndarray, 
                check_singular: bool = True) -> np.ndarray:
    """
    Form inverse via singular value decomposition (SVD). Not recommended as 
    numerically unstable.

    Parameters
    ----------
    A : np.ndarray shape (k,k) or (n,k,k)
        Matrix/matrices to invert.
    check_singular : bool
        Check if singular. If not True, raise ValueError.

    Returns
    -------
    Ainv : np.ndarray shape (k,k) or (n,k,k)
        inverted matrix/matrices.
    """
    #
    warn("Forming a matrix inverse is generally a bad idea. ")
    #
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    if check_singular:
        if np.isclose(s,0.).any():
            raise ValueError("A is singular or close to singular. Inversion unstable/impossible.")
    #
    if len(s.shape)==2:
        return np.einsum('ibj,ij,iaj->iab', 
                         Vt.transpose(0,2,1), 1/s, U)
    elif len(s.shape)==1:
        return np.einsum('bj,j,aj->ab', 
                         Vt.T, 1/s, U)
    
def solve_inverse(A: np.ndarray) -> np.ndarray:
    """
    Form inverse via solving:
        
        solve(A,eye)

    Parameters
    ----------
    A : np.ndarray shape (k,k) or (n,k,k)
        Matrix/matrices to invert.

    Returns
    -------
    Ainv : np.ndarray shape (k,k) or (n,k,k)
        inverted matrix/matrices.
    """
    return np.linalg.solve(A, np.eye(A.shape[-1]))

def cholesky_inverse(A: np.ndarray) -> np.ndarray:
    """
    Form inverse via Cholesky decomposition:
        
        A = L @ L.T

    Parameters
    ----------
    A : np.ndarray shape (k,k) or (n,k,k)
        Matrix/matrices to invert.

    Returns
    -------
    Ainv : np.ndarray shape (k,k) or (n,k,k)
        inverted matrix/matrices.
    """
    L = np.linalg.cholesky(A)
    if len(A.shape) == 3:
        return solve_triangular(L.transpose((0,2,1)),
                                solve_triangular(L,
                                                 np.eye(A.shape[-1]),
                                                 lower=True),
                                lower=False)
    elif len(A.shape) == 2:
        return solve_triangular(L.T, 
                                solve_triangular(L,
                                                 np.eye(A.shape[-1]),
                                                 lower=True),
                                lower=False)
