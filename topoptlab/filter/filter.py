from abc import ABC, abstractmethod

import numpy as np

class TOFilter(ABC):
    """
    Base class for all filters and projections that allows modular stacking 
    of filters.  
    """
    
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
    def apply_filter(self, x: np.ndarray) -> np.ndarray:
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
                        x_filtered : np.ndarray, 
                        dx_filtered : np.ndarray) -> np.ndarray:
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
        ...