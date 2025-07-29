from typing import Callable, Tuple, Union, Any

import numpy as np
from scipy.ndimage import convolve

def assemble_convolution_filter(nelx: int, nely: int, rmin: float,
                                mapping: Callable, invmapping: Callable,
                                nelz: Union[int, None] = None,
                                **kwargs: Any) -> Tuple[np.ndarray,np.ndarray]:
    """
    Assemble distance based filter as image/voxel convolution filter. Returns
    the kernel and the normalization constants

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    rmin : float
        cutoff radius for the filter. Only elements within the element-center
        to element center distance are used for filtering.
    mapping : callable,
        converts property from 1D np.ndarray to image/voxel.
    invmapping : callable,
        converts property from image/voxel to 1D np.ndarray in correct order.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    h : np.ndarray, shape (nfilter,nfilter) or (nfilter,nfilter,nfilter)
        convolution kernel.
    hs : np.ndarray, shape (n)
        normalization constants.

    """
    # filter radius in number of elements
    nfilter = int(2*np.floor(rmin)+1)
    #
    x = np.arange(-np.floor(rmin),np.floor(rmin)+1)
    if nelz is None:
        #
        n = nelx*nely
        #
        x = np.tile(x,(nfilter,1))
        y = np.rot90(x)
        # hat function
        kernel = np.maximum(0.0,rmin - np.sqrt(x**2 + y**2))
    else:
        #
        n = nelx*nely*nelz
        #
        x = np.tile(x,(nfilter,nfilter,1))
        y = x.transpose((0,2,1))
        z = x.transpose((2,1,0))
        # hat function
        kernel = np.maximum(0.0,rmin - np.sqrt(x**2 + y**2 + z**2))
    # normalization constants
    hs = invmapping(convolve(mapping(np.ones(n ,dtype=np.float64)),
                             kernel,
                             mode="constant",
                             cval=0))
    return kernel,hs[:,None]