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
                 nelx: int, nely: int, rmin: float,
                 nelz: Union[int, None] = None,
                 filter_mode: str = "matrix",
                 **kwargs: Any) -> None:
        """
        Initialize filter and construct the filter if necessary
        
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
        filter_mode : str
            indicates how filtering is done. Possible values are "matrix" or
            "helmholtz". If "matrix", then density/sensitivity filters are
            implemented via a sparse matrix and applied by multiplying
            said matrix with the densities/sensitivities.
        
        Returns
        -------
        None

        """
        if filter_mode == "matrix":
            self.filter = MatrixFilter(nelx=nelx, 
                                       nely=nely, 
                                       rmin=rmin,
                                       nelz=nelz)
        elif filter_mode == "helmholtz":
            self.filter = HelmholtzFilter(nelx=nelx, 
                                          nely=nely, 
                                          rmin=rmin,
                                          nelz=nelz)  
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
            (intermediate) design variables.

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
        x_filtered : np.ndarray
            filtered design variables.
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