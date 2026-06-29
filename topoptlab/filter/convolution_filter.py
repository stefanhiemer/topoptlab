# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Tuple, Union, Any
from functools import partial

import numpy as np
from scipy.ndimage import convolve

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.kernels import hat_kernel
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
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 rmin: float,
                 kernel_fn: Callable,
                 nelz: Union[int, None] = None,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 **kwargs: Any) -> None:
        """
        Assemble convolution-based filter from "Efficient topology optimization
        in MATLAB using 88 lines of code".

        Parameters
        ----------
        nelx : int
            number of elements in x direction.
        nely : int
            number of elements in y direction.
        n_constr : int
            number of constraints.
        rmin : float
            cutoff radius for the filter.
        kernel_fn : callable
            weighting kernel passed to the underlying filter assembly.
        nelz : int or None
            number of elements in z direction.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
            Otherwise, a boolean array indicating which constraint
            sensitivities are filtered.

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
                                                      invmapping=self.invmapping,
                                                      kernel_fn=kernel_fn)
        self._filter_objective = filter_objective
        if constraint_filter_mask is None:
            self._constraint_filter_mask = np.ones(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
                constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        else:
            raise TypeError("constraint_filter_mask must be None or np.ndarray of shape (n_constr,).")

    def apply_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply filter to the (intermediate) design variables x:
            
            x_filtered = np.asarray(H*(dobj/Hs))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables, shape (n, k).

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables, shape (n, k).

        """
        return self.invmapping(convolve(self.mapping(x),
                                        weights=self.h, 
                                        axes=(0,1,2)[:self.ndim],
                                        mode="constant",
                                        cval=0.)) / self.hs
        
    def apply_filter_dx(self, 
                        dx_filtered: np.ndarray, 
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = H @ (dx_filtered / Hs)
        
        Parameters
        ----------
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables,
            shape (n, k).

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables,
            shape (n, k).
        """
        return self.invmapping(convolve(self.mapping(dx_filtered / self.hs),
                                        weights=self.h,
                                        axes=(0,1,2)[:self.ndim],
                                        mode="constant",
                                        cval=0.0))

    @property
    def vol_conserv(self) -> bool:
        """
        Returns True as the convolution filter is volume conserving.
        """
        return True

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.
        """
        return self._filter_objective

    @property
    def constraint_filter_mask(self) -> np.ndarray:
        """
        Boolean array of shape (n_constr,) indicating which constraint
        sensitivities the filter is applied to.

        Returns
        -------
        constraint_filter_mask : np.ndarray of shape (n_constr,)
        """
        return self._constraint_filter_mask
    
    @property
    def changes_filter_kw(self) -> bool:
        return False

    def update_filter_kw(self, filter_kw: dict) -> None:
        return


def assemble_convolution_filter(nelx: int, 
                                nely: int, 
                                rmin: float,
                                mapping: Callable,
                                invmapping: Callable,
                                kernel_fn: Callable,
                                nelz: Union[int, None] = None,
                                compute_coords: bool = True,
                                **kwargs: Any) -> Tuple[np.ndarray,np.ndarray]:
    """
    Assemble distance based filter as image/voxel convolution filter. Returns
    the kernel and the normalization constants.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    rmin : float
        cutoff radius for the filter. Only elements within the element-center
        to element center distance are used for filtering.
    mapping : callable
        converts property from 1D np.ndarray to image/voxel.
    invmapping : callable
        converts property from image/voxel to 1D np.ndarray in correct order.
    kernel_fn : callable
        function that returns the convolution kernel. When compute_coords is
        True (default), called as kernel_fn(x, y, rmin) in 2D or
        kernel_fn(x, y, rmin, z) in 3D with integer offset grids and rmin.
        When compute_coords is False, called as kernel_fn(rmin) and must
        return the kernel array directly.
    nelz : int or None
        number of elements in z direction.
    compute_coords : bool
        if True (default), coordinate offset grids are computed and passed to
        kernel_fn. If False, kernel_fn is called with rmin only.

    Returns
    -------
    h : np.ndarray, shape (nfilter,nfilter) or (nfilter,nfilter,nfilter)
        convolution kernel.
    hs : np.ndarray, shape (n,)
        normalization constants.

    """
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    n = np.prod([nelx,nely,nelz][:ndim]).astype(int)
    nfilter = int(2*np.floor(rmin)+1)
    #
    if compute_coords:
        offsets = np.arange(-np.floor(rmin), np.floor(rmin)+1)
        if nelz is None:
            x = np.tile(offsets, (nfilter, 1))
            y = np.rot90(x)
            kernel = kernel_fn(x=x, y=y, 
                               z=None, 
                               rmin=rmin)
        else:
            x = np.tile(offsets, (nfilter, nfilter, 1))
            y = x.transpose((0, 2, 1))
            z = x.transpose((2, 1, 0))
            kernel = kernel_fn(x=x, y=y, rmin=rmin, z=z)
    else:
        kernel = kernel_fn(rmin=rmin)
    # normalization constants
    hs = invmapping(convolve(mapping(np.ones(n, dtype=np.float64)),
                             kernel,
                             axes=(0,1,2)[:ndim],
                             mode="constant",
                             cval=0))

    return kernel, hs[:,None]