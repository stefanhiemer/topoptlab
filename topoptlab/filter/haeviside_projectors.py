# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.haeviside_projection import find_eta

class HaevisideProjectorGuest2004(TOFilter):
    """
    
    Implements the Haeviside projection by 
    
    Guest, James K., Jean H. Prévost, and Ted Belytschko. "Achieving minimum 
    length scale in topology optimization using nodal design variables and 
    projection functions." International journal for numerical methods in 
    engineering 61.2 (2004): 238-254.
    
    This projection is a smooth version of the Haeviside step function Theta(x), 
    so in simple words, this projection sets every value that is larger than 
    zero to one and everything smaller/equal to zero to zero. The filter 
    equation is 
    
    x_filtered = 1 - exp(-beta x) + x exp(-beta)
    
    beta is the projection strength, that is typically ramped up during the TO 
    process to large values. The larger beta, the closer this filter is to a 
    Haeviside function. 
    """
    
    def __init__(self,
                 **kwargs: Any) -> None:
        """
        Do nothing as this filter requires no initialization.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None

        """
        return
        
    def apply_filter(self, 
                     x: np.ndarray, 
                     beta=float,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x
        
        x_filtered = 1 - exp(-beta x) + x exp(-beta)
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        beta : float
            projection strength.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return 1 - np.exp(-beta*x) + x*np.exp(-beta)
    
    def apply_filter_dx(self, 
                        x_filtered : np.ndarray, 
                        beta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming 
        
            x_filtered = 1 - exp(-beta x) + x exp(-beta) 
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = beta*np.exp(-beta*x_filtered) + np.exp(-beta)
        
        Parameters
        ----------
        x_filtered : np.ndarray
            filtered design variables.
        beta : float
            projection strength.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return beta*np.exp(-beta*x_filtered) + np.exp(-beta)
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to False as filter is not volume conserving. 
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        False
        """
        return False

class HaevisideProjectorSigmund2007(TOFilter):
    """
    
    Implements the Haeviside projection by 
    
    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.
    
    This projection is a different flavor of the Guest2004 projection and 
    approximates the shifted Haeviside step function Theta(1-x), so in this 
    projection sets every value that is smaller than one to zero and everything 
    smaller/equal to one to one. The filter equation is 
    
    x_filtered = np.exp(-beta*(1-x)) - (1-x)*np.exp(-beta)
    
    beta is the projection strength, that is typically ramped up during the TO 
    process to large values. The larger beta, the closer this filter is to a 
    shifted Haeviside function. 
    """
    
    def __init__(self,
                 **kwargs: Any) -> None:
        """
        Do nothing as this filter requires no initialization.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None

        """
        return
        
    def apply_filter(self, 
                     x: np.ndarray, 
                     beta=float,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x
        
        x_filtered = np.exp(-beta*(1-x)) - (1-x)*np.exp(-beta)
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        beta : float
            projection strength.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)
    
    def apply_filter_dx(self, 
                        x_filtered : np.ndarray, 
                        beta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming 
        
            x_filtered = np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = beta*np.exp(-beta*x_filtered) + np.exp(-beta)
        
        Parameters
        ----------
        x_filtered : np.ndarray
            filtered design variables.
        beta : float
            projection strength.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return np.exp(beta*(x_filtered-1)) * beta + np.exp(-beta)
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to False as filter is not volume conserving. 
        
        Parameters
        ----------
        None.
            
        Returns
        -------
        False
        """
        return False
    
class EtaProjectorXu2010(TOFilter):
    """
    
    Implements the Haeviside projection by 
    
    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    This projection generalizes the projections by Guest2004 and Sigmund2007 by
    smoothly approximating the Haeviside function Theta(x-eta) with the 
    projection threshold eta, so every value smaller than eta is set to zero 
    and every value smaller/equal to eta set to one. The filter equation is 
    
    x_filtered = (tanh(beta*eta)+tanh(beta * (x - eta)))/\ 
                 (tanh(beta*eta)+tanh(beta*(1-eta)))
    
    is the projection strength, that is typically ramped up during the TO 
    process to large values. The larger beta, the closer this filter is to a 
    shifted Haeviside function. 
    """
    
    def __init__(self,
                 vol_conserving,
                 **kwargs: Any) -> None:
        """
        Initialize filter by setting volume conserving flag.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None

        """
        self._vol_conserv = vol_conserving
        return
        
    def apply_filter(self, 
                     x : np.ndarray, 
                     beta : float,
                     eta : float,
                     volfrac : Union[None,float],
                     root_args : Union[None,float],
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x
        
        x_filtered = (tanh(beta*eta)+tanh(beta * (x - eta)))/\ 
                     (tanh(beta*eta)+tanh(beta*(1-eta)))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        beta : float
            projection strength.
        eta : float
            projection threshold.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        
        if self.vol_conserv:
            eta = find_eta(eta0 = eta, 
                           xTilde = x, 
                           beta = beta, 
                           volfrac = volfrac,
                           root_args = root_args)
        return (np.tanh(beta*eta)+np.tanh(beta * (x - eta)))/\
                (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
    
    def apply_filter_dx(self, 
                        x_filtered : np.ndarray, 
                        beta : float,
                        eta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming 
        
            x_filtered = np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = beta * (1 - tanh(beta * (x_filtered - eta))**2) /\
                    (tanh(beta*eta)+tanh(beta*(1-eta)))
        
        Parameters
        ----------
        x_filtered : np.ndarray
            filtered design variables.
        beta : float
            projection strength.
        eta : float
            projection threshold.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return beta * (1 - np.tanh(beta * (x_filtered - eta))**2) /\
                (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to  as filter is not volume conserving. 
        
        Parameters
        ----------
        vol_conserving : bool
            True, if filter is vol. conserving by finding the right eta. 
            
        Returns
        -------
        vol_conserv : bool
            True if filter is volume conserving.
        """
        return self._vol_conserv
    
