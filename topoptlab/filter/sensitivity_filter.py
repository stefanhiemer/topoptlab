# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.matrix_filter import MatrixFilter
from topoptlab.filter.helmholtz_filter import HelmholtzFilter
from topoptlab.filter.convolution_filter import ConvolutionFilter

class SensitivityFilter(TOFilter):
    """
    
    Implements the sensitivity filter by 
    
    Sigmund, Ole. "On the design of compliant mechanisms using topology 
    optimization." Journal of Structural Mechanics 25.4 (1997): 493-524.
    
    and also its extension to a PDE based filter by 
    
    Lazarov, Boyan Stefanov, and Ole Sigmund. "Filters in topology optimization 
    based on Helmholtz‐type differential equations." International journal for 
    numerical methods in engineering 86.6 (2011): 765-781.
    """
    
    def __init__(self,
                 nelx: int, nely: int, rmin: float,
                 nelz: Union[int, None] = None,
                 filter_mode: str = "matrix",
                 gamma: float = 1e-3,
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
        gamma : float
            small value to avoid zero division.
        
        Returns
        -------
        None

        """
        #
        self.gamma = gamma
        #
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
        
        x_filtered = filter(x)
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return x
    
    def apply_filter_dx(self, 
                        x_filtered : np.ndarray, 
                        dx_filtered : np.ndarray,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
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
        return self.filter.apply_filter(x_filtered=None,
                                        dx_filtered=dx_filtered) / \
               np.maximum(self.gamma, x_filtered)
    
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