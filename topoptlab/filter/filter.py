# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Union
from abc import ABC, abstractmethod

import numpy as np

class TOFilter(ABC):
    """
    Base class for all filters and projections that allows modular stacking 
    of filters.  
    """
    vol_conserv : bool 
    
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize filter and construct the filter if necessary
        
        Returns
        -------
        None

        """
        ...
    
    @abstractmethod
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
        ...
    
    @abstractmethod
    def apply_filter_dx(self, 
                        x : np.ndarray,
                        x_filtered : np.ndarray, 
                        dx_filtered : np.ndarray,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables.
        Not for all types of filters either x_filtered or dx_filtered are 
        needed.
        
        Parameters
        ----------
        x : np.ndarray
            unfiltered variables.
        x_filtered : np.ndarray
            filtered  variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        ...
    
    @property
    @abstractmethod
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to indicate if filter is volume conserving. 
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        vol_conserv : bool
            True if filter is volume conserving.
            
        """
        ...
    
    @property
    @abstractmethod
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to indicate if filter is volume conserving. 
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        vol_conserv : bool
            True if filter is volume conserving.
            
        """
        ...
        
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
        ...
    
    @property
    def constraint_filter_mask(self) -> Union[bool,np.ndarray]:
        """
        Indicate if filter is applied to constraint sensitivities.

        Parameters
        ----------
        None.

        Returns
        -------
        constraint_filter_mask : bool
            True if filter is applied to all constraint sensitivities,
            False if none are filtered, or a boolean mask indicating
            which constraint sensitivities are filtered.

        """
        ...

    @property
    @abstractmethod
    def changes_filter_kw(self) -> bool:
        """
        True if this filter writes state back to filter_kw after each forward
        pass (e.g. an updated eta / etas).

        Returns
        -------
        changes_filter_kw : bool
        """
        ...

    @abstractmethod
    def update_filter_kw(self, filter_kw: dict) -> None:
        """
        Write any filter state needed by downstream stages or continuation
        into ``filter_kw`` in-place.

        Parameters
        ----------
        filter_kw : dict
            filter keyword dictionary, updated in-place.
        """
        ...