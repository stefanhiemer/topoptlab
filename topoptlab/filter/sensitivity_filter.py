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
                 nelx: int,
                 nely: int,
                 n_constr : int,
                 rmin: float,
                 nelz: Union[int, None] = None,
                 filter_mode: str = "matrix",
                 gamma: float = 1e-3,
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
        gamma : float
            small value to avoid zero division.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : bool
            True if filter is applied to all constraint sensitivities,
            False if none are filtered, or a boolean mask indicating
            which constraint sensitivities are filtered.
        el_flags : None or np.ndarray
            array of element flags (0 free, 1 passive, 2 active).
        el_flags_policy : None or dict
            policy dict controlling filter behaviour for prescribed elements.

        Returns
        -------
        None

        """
        #
        self.gamma = gamma
        #
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
            self._constraint_filter_mask = np.zeros(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
            constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        elif isinstance(constraint_filter_mask, np.ndarray) and \
            constraint_filter_mask.shape != (n_constr,):
            raise TypeError("constraint_filter_mask have shape (n_constr,): ", constraint_filter_mask.shape)
        else:
            raise TypeError("constraint_filter_mask must be None or np.ndarray of shape (n_constr,): ", type(constraint_filter_mask))
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
            unfiltered (design) variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return x
    
    def apply_filter_dx(self, 
                        x : np.ndarray, 
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
        x : np.ndarray
            unfiltered (design) variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return self.filter.apply_filter(x*dx_filtered) / np.maximum(self.gamma, x)
    
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
    def constraint_filter_mask(self) -> Union[bool,np.ndarray]:
        """
        Indicate if filter is applied to constraint sensitivities.

        Returns
        -------
        constraint_filter_mask : bool
        """
        return self._constraint_filter_mask

    @property
    def changes_filter_kw(self) -> bool:
        return False

    def update_filter_kw(self, filter_kw: dict) -> None:
        return