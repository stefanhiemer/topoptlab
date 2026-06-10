# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Dict, Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.haeviside_projection import find_eta, eta_projection, eta_projection_dx,\
                                                  find_multieta, multieta_projection, multieta_projection_dx,\
                                                  guest_projection, guest_projection_dx,\
                                                  sigmund2007_projection, sigmund2007_projection_dx

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
                 n_constr: int,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 **kwargs: Any) -> None:
        """
        Initialize filter.

        Parameters
        ----------
        n_constr : int
            number of constraints.
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
                     beta: float,
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
        return guest_projection(x=x, beta=beta)
    
    def apply_filter_dx(self,
                        x : np.ndarray,
                        dx_filtered : np.ndarray,
                        beta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables
        x_filtered using the chain rule assuming

            x_filtered = 1 - exp(-beta x) + x exp(-beta)

        to get the sensitivities with respect to the (unfiltered) design
        variables or in the case of many filters intermediate design variables:

            dx = dx_filtered * (beta*exp(-beta*x) + exp(-beta))

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
        beta : float
            projection strength.

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return dx_filtered * guest_projection_dx(x=x, beta=beta)
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to False as filter is not volume conserving.

        Returns
        -------
        False
        """
        return False

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.

        Returns
        -------
        filter_objective : bool
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

class HaevisideProjectorSigmund2007(TOFilter):
    """
    
    Implements the Haeviside projection by 
    
    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.
    
    This projection is a different flavor of the Guest2004 projection and 
    approximates the shifted Haeviside step function Theta(1-x), so this 
    projection sets every value that is smaller than one to zero and everything 
    smaller/equal to one to one. The filter equation is 
    
    x_filtered = np.exp(-beta*(1-x)) - (1-x)*np.exp(-beta)
    
    beta is the projection strength, that is typically ramped up during the TO 
    process to large values. The larger beta, the closer this filter is to a 
    shifted Haeviside function. 
    """
    
    def __init__(self,
                 n_constr: int,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 **kwargs: Any) -> None:
        """
        Initialize filter.

        Parameters
        ----------
        n_constr : int
            number of constraints.
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
                     beta: float,
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
        return sigmund2007_projection(x=x, beta=beta)
    
    def apply_filter_dx(self,
                        x : np.ndarray,
                        dx_filtered : np.ndarray,
                        beta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables
        x_filtered using the chain rule assuming

            x_filtered = np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)

        to get the sensitivities with respect to the (unfiltered) design
        variables or in the case of many filters intermediate design variables:

            dx = dx_filtered * (beta*exp(beta*(x-1)) + exp(-beta))

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
        beta : float
            projection strength.

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return dx_filtered * sigmund2007_projection_dx(x=x, beta=beta)
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to False as filter is not volume conserving.

        Returns
        -------
        False
        """
        return False

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.

        Returns
        -------
        filter_objective : bool
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


class EtaProjectorXu2010(TOFilter):
    """
    
    Implements the Haeviside projection by 
    
    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    This projection generalizes the projections by Guest2004 and Sigmund2007 by
    smoothly approximating the Haeviside function Theta(x-eta) with the 
    projection threshold eta, so every value smaller than eta is set to zero 
    and every value smaller/equal to eta set to one. The filter equation is 
    
    x_filtered = (tanh(beta*eta)+tanh(beta * (x - eta)))/
                 (tanh(beta*eta)+tanh(beta*(1-eta)))
    
    is the projection strength, that is typically ramped up during the TO 
    process to large values. The larger beta, the closer this filter is to a 
    shifted Haeviside function. 
    """
    
    def __init__(self,
                 n_constr: int,
                 volfrac: Union[None, float],
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 eta: Union[None, float] = None,
                 adapt_eta: bool = True,
                 **kwargs: Any) -> None:
        """
        Initialize filter.

        Parameters
        ----------
        n_constr : int
            number of constraints.
        volfrac : None or float
            target volume fraction used to initialise eta when eta is None.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
            Otherwise, a boolean array indicating which constraint
            sensitivities are filtered.
        eta : None or float
            initial projection threshold. If None, volfrac is used as the
            initial value.
        adapt_eta : bool
            if True (default), eta is updated each call to apply_filter via a
            root search to preserve the volume fraction, making the filter
            volume conserving. If False, eta stays fixed.

        Returns
        -------
        None

        """
        if eta is not None:
            self.eta = eta
        elif volfrac is not None:
            self.eta = volfrac
        else:
            raise ValueError("Either eta or volfrac must be provided to initialise the threshold.")
        self.adapt_eta = adapt_eta
        self._vol_conserv = adapt_eta
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
                     x : np.ndarray, 
                     beta : float,
                     volfrac : Union[None,float],
                     root_args : union[None,Dict] = None,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x
        
        x_filtered = (tanh(beta*eta)+tanh(beta * (x - eta)))/
                     (tanh(beta*eta)+tanh(beta*(1-eta)))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        beta : float
            projection strength.
        volfrac : None or float
            target volume fraction. If not None, eta is updated via a root
            search before applying the projection.
        root_args : None or dict
            keyword arguments passed to find_eta for the root search. If None, defaults
            if find_deta are used.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """

        if self.adapt_eta and volfrac:
            if root_args is None:
                kw = {} 
            else: 
                kw = {"root_args": root_args}
            self.eta = find_eta(eta0 = self.eta,
                                xTilde = x,
                                beta = beta,
                                volfrac = volfrac,
                                **kw)
        return eta_projection(eta=self.eta, 
                              xTilde=x, 
                              beta=beta)
    
    def apply_filter_dx(self,
                        x : np.ndarray,
                        dx_filtered : np.ndarray,
                        beta : float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables
        x_filtered using the chain rule assuming

            x_filtered = (tanh(beta*eta) + tanh(beta*(x - eta))) /
                         (tanh(beta*eta) + tanh(beta*(1 - eta)))

        to get the sensitivities with respect to the (unfiltered) design
        variables or in the case of many filters intermediate design variables:

            dx = dx_filtered * beta * (1 - tanh(beta*(x - eta))**2) /
                    (tanh(beta*eta) + tanh(beta*(1 - eta)))

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
        beta : float
            projection strength.

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return eta_projection_dx(eta=self.eta,
                                 xTilde=x,
                                 beta=beta)*dx_filtered
    
    @property
    def vol_conserv(self) -> bool:
        """
        True if filter is volume conserving (eta found via root search).

        Returns
        -------
        vol_conserv : bool
        """
        return self._vol_conserv

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.

        Returns
        -------
        filter_objective : bool
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
        return self.adapt_eta

    def update_filter_kw(self, filter_kw: dict) -> None:
        if self.adapt_eta:
            filter_kw["eta"] = self.eta

class MultiEtaProjectorXu2010(TOFilter):
    """
    Multi-threshold Heaviside projection based on

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495-505

    Applies N independent projections at thresholds ``etas`` and combines
    them into a single scalar field via a weighted sum:

        x_filtered = sum_n w_n * H_{beta, eta_n}(x)

    where each H_{beta, eta_n} is the single-threshold tanh projection.
    This allows smooth multi-phase interpolation from a single design field.

    When ``adapt_eta=True`` (default) and ``volfrac`` is provided, the
    thresholds are updated each call via ``find_multieta`` (MSE-minimising,
    volume-constrained optimisation) to keep the projected volume fraction
    close to the target. Set ``adapt_eta=False`` to keep thresholds fixed.
    """

    def __init__(self,
                 n_constr: int,
                 volfrac: Union[None, float],
                 n_etas: int,
                 weights: Union[None, np.ndarray] = None,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 etas: Union[None, np.ndarray] = None,
                 adapt_eta: bool = True,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_constr : int
            number of constraints.
        volfrac : None or float
            target volume fraction passed to ``find_multieta`` when
            ``adapt_eta`` is True.
        n_etas : int
            number of threshold values.
        weights : None or np.ndarray of shape (n_etas,)
            combination weights. Uniform weights are used when None.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
        etas : None or np.ndarray of shape (n_etas,)
            initial threshold values. Defaults to n_etas equally spaced
            values in (0, 1) when None.
        adapt_eta : bool
            if True (default), ``etas`` are updated each call to
            ``apply_filter`` via ``find_multieta``, making the filter volume
            conserving. If False, ``etas`` stay fixed.
        """
        if etas is None:
            self.etas = np.linspace(0.0, 1.0, n_etas + 2)[1:-1]
        else:
            self.etas = np.asarray(etas, dtype=float)
        if weights is None:
            self.weights = np.ones(n_etas) / n_etas
        else:
            self.weights = np.asarray(weights, dtype=float)
        self.adapt_eta = adapt_eta
        self._vol_conserv = adapt_eta
        self._filter_objective = filter_objective
        if constraint_filter_mask is None:
            self._constraint_filter_mask = np.ones(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
                constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        else:
            raise TypeError("constraint_filter_mask must be None or np.ndarray of shape (n_constr,).")

    def apply_filter(self,
                     x: np.ndarray,
                     beta: float,
                     volfrac: Union[None, float] = None,
                     find_multieta_kw: Dict = {"mode": "mse"},
                     **kwargs: Any) -> np.ndarray:
        """
        Apply the multi-threshold projection to ``x``.

        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        beta : float
            projection strength.
        volfrac : None or float
            target volume fraction. If not None, ``self.etas`` are updated
            via ``find_multieta`` before projecting.
        find_multieta_kw : dict
            keyword arguments forwarded to ``find_multieta``.

        Returns
        -------
        x_filtered : np.ndarray
            projected design variables.
        """
        if self.adapt_eta and volfrac is not None:
            self.etas = find_multieta(etas0=self.etas,
                                      xTilde=x.ravel(),
                                      beta=beta,
                                      volfrac=volfrac,
                                      weights=self.weights,
                                      **find_multieta_kw)
        return multieta_projection(etas=self.etas,
                                   xTilde=x,
                                   beta=beta,
                                   weights=self.weights)

    def apply_filter_dx(self,
                        x: np.ndarray,
                        dx_filtered: np.ndarray,
                        beta: float,
                        **kwargs: Any) -> np.ndarray:
        """
        Chain-rule pullback of sensitivities through the multi-threshold
        projection.

        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to the projected field.
        beta : float
            projection strength.

        Returns
        -------
        dx : np.ndarray
            sensitivities with respect to ``x``.
        """
        return multieta_projection_dx(etas=self.etas,
                                      xTilde=x,
                                      beta=beta,
                                      weights=self.weights) * dx_filtered

    @property
    def vol_conserv(self) -> bool:
        return self._vol_conserv

    @property
    def filter_objective(self) -> bool:
        return self._filter_objective

    @property
    def constraint_filter_mask(self) -> np.ndarray:
        return self._constraint_filter_mask

    @property
    def changes_filter_kw(self) -> bool:
        return self.adapt_eta

    def update_filter_kw(self, filter_kw: dict) -> None:
        if self.adapt_eta:
            filter_kw["etas"] = self.etas


def vol_conserv():

    return 