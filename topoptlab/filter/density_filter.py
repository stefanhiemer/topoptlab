# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.matrix_filter import MatrixFilter
from topoptlab.filter.helmholtz_filter import HelmholtzFilter
from topoptlab.filter.convolution_filter import ConvolutionFilter

class DensityFilter(TOFilter):
    """
    
    Implements the density filter by 
    
    Bruns, Tyler E., and Daniel A. Tortorelli. "Topology optimization of 
    non-linear elastic structures and compliant mechanisms." Computer methods 
    in applied mechanics and engineering 190.26-27 (2001): 3443-3459.
    
    and also its extension to a PDE based filter by 
    
    Lazarov, Boyan Stefanov, and Ole Sigmund. "Filters in topology optimization 
    based on Helmholtz‐type differential equations." International journal for 
    numerical methods in engineering 86.6 (2011): 765-781.
    
    The filter in its most basic form is just a blurring filter as 
    conventionally used in image manipulation and can be written as a 
    convolution integral
    
    x_filtered = int K(r,s) x dr
    
    where K(r,s) is the convolution kernel. The commonly kernel used kernel is 
    hat function, but other variants exist as well. This convolution is 
    implemented via the standard matrix filter or the PDE filter as described 
    in 
    
    Andreassen, Erik, et al. "Efficient topology optimization in MATLAB using 
    88 lines of code." Structural and Multidisciplinary Optimization 43.1 (
    2011): 1-16.
    
    and a prototype using scipy's ndimage convolution is on the way, but not 
    yet working.
    """
    
    def __init__(self,
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 rmin: float,
                 nelz: Union[int, None] = None,
                 filter_mode: str = "matrix",
                 filter_objective : bool = True,
                 constraint_filter_mask : Union[None,np.ndarray] = None,
                 el_flags: Union[None, np.ndarray] = None,
                 el_flags_policy: Union[None, dict] = None,
                 l: np.ndarray = np.array([1., 1.]),
                 **kwargs: Any) -> None:
        """
        Initialize filter and construct the filter if necessary

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
        nelz : int or None
            number of elements in z direction.
        filter_mode : str
            indicates how filtering is done. Possible values are "matrix" or
            "helmholtz". If "matrix", then density/sensitivity filters are
            implemented via a sparse matrix and applied by multiplying
            said matrix with the densities/sensitivities.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
            Otherwise, a boolean array indicating which constraint
            sensitivities are filtered.
        el_flags : None or np.ndarray
            array of element flags (0 free, 1 passive, 2 active).
        el_flags_policy : None or dict
            policy dict controlling filter behaviour for prescribed elements.

        Returns
        -------
        None

        """
        if el_flags_policy is not None and el_flags_policy["neglect_in_filter"] and \
                filter_mode != "matrix":
            raise NotImplementedError("neglect_in_filter is only supported for filter_mode='matrix'.")
        if filter_mode == "matrix":
            self.filter = MatrixFilter(nelx=nelx,
                                       nely=nely,
                                       n_constr=n_constr,
                                       rmin=rmin,
                                       nelz=nelz,
                                       el_flags=el_flags,
                                       el_flags_policy=el_flags_policy)
        elif filter_mode == "helmholtz":
            self.filter = HelmholtzFilter(nelx=nelx,
                                          nely=nely,
                                          n_constr=n_constr,
                                          rmin=rmin,
                                          nelz=nelz,
                                          l=l)
        #
        self._filter_objective = filter_objective
        if constraint_filter_mask is None:
            self._constraint_filter_mask = np.ones(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
                constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        else:
            raise TypeError("constraint_filter_mask must be None or np.ndarray of shape (n_constr,).")
        return
        
    def apply_filter(self, 
                     x: np.ndarray,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x
        
        x_filtered = H@x / Hs 
        
        Parameters
        ----------
        x : np.ndarray
            unfiltered variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return self.filter.apply_filter(x=x)
    
    def apply_filter_dx(self, 
                        dx_filtered : np.ndarray,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = H@x / Hs 
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables. 
        
        Parameters
        ----------
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return self.filter.apply_filter_dx(x_filtered=None,
                                           dx_filtered=dx_filtered)
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to True as filter is volume conserving. 
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        True
        """
        return True

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
            
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