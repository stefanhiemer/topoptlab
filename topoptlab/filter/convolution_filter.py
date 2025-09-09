# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Tuple, Union, Any
from functools import partial

import numpy as np
from scipy.ndimage import convolve

from topoptlab.filter.filter import TOFilter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel

class ConvolutionFilter(TOFilter):
    """
    Implementation here is based on the implementation in 
    
    Andreassen, Erik, et al. "Efficient topology optimization in MATLAB using 
    88 lines of code." Structural and Multidisciplinary Optimization 43.1 
    (2011): 1-16.
    
    but extended to 3D.
    """
    
    def __init__(self,
                 nelx: int, nely: int, rmin: float,
                 nelz: Union[int, None] = None, 
                 **kwargs: Any) -> None:
        """
        Assemble matrix-based filter from "Efficient topology optimization in 
        MATLAB using 88 lines of code".
        
        Parameters
        ----------
        nelx : int
            number of elements in x direction.
        nely : int
            number of elements in y direction.
        rmin : float
            cutoff radius for the filter.
        nelz : int or None
            number of elements in z direction.
        
        Returns
        -------
        None
        """
        #
        if nelz is None:
            self.mapping = partial(map_eltoimg,
                                   nelx=nelx,
                                   nely=nely)
            self.invmapping = partial(map_imgtoel,
                                      nelx=nelx,
                                      nely=nely)
            self.ndim = 2
        else:
            self.mapping = partial(map_eltovoxel,
                                   nelx=nelx,
                                   nely=nely,
                                   nelz=nelz)
            self.invmapping = partial(map_voxeltoel,
                                      nelx=nelx,
                                      nely=nely,
                                      nelz=nelz)
            self.ndim = 3
        #
        self.h, self.hs = assemble_convolution_filter(nelx=nelx,
                                                      nely=nely,
                                                      nelz=nelz,
                                                      rmin=rmin,
                                                      mapping=self.mapping,
                                                      invmapping=self.invmapping)
        
    def apply_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply filter to the (intermediate) design variables x:
            
            x_filtered = np.asarray(H*(dobj/Hs))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return self.invmapping(convolve(self.mapping(x),
                                        weights=self.h, 
                                        axes=(0,1,2)[:self.ndim],
                                        mode="constant",
                                        cval=0.0)) / self.hs
        
    def apply_filter_dx(self, dx_filtered: np.ndarray) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = H@dx_filtered / Hs
        
        Parameters
        ----------
        x_filtered : np.ndarray
            filtered design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return self.invmapping(convolve(self.mapping(dx_filtered),
                                        weights=self.h, 
                                        axes=(0,1,2)[:self.ndim],
                                        mode="constant",
                                        cval=0.0)) / self.hs

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