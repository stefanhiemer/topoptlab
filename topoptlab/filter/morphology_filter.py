# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.kernels import uniform_kernel
from topoptlab.filter.matrix_filter import MatrixFilter
from topoptlab.filter.helmholtz_filter import HelmholtzFilter
from topoptlab.filter.convolution_filter import ConvolutionFilter

class DilateFilterSigmund2007(TOFilter):
    """
    Implements the dilate filter by

    Sigmund, Ole. "Morphology-based black and white filters for topology
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007):
    401-424.

    The dilate filter is a smooth approximation of the maximum over the
    neighborhood of an element, so solid regions grow. The filter equation is

    x_filtered = log( F(exp(beta*x)) ) / beta

    where F is the normalized neighborhood average. In the original work the
    neighborhood is weighted uniformly within rmin which is used for
    filter_mode "matrix" and "convolution". For "helmholtz" the implicit
    kernel of the PDE filter replaces the uniform neighborhood. For numerical
    stability it is evaluated as

    x_filtered = 1 + log( F(exp(beta*(x-1))) ) / beta.

    beta is the filter strength, that is typically ramped up during the TO
    process to large values. For beta to zero the filter reduces to the
    density filter, for beta to infinity to the maximum operator.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 rmin: float,
                 nelz: int | None = None,
                 filter_mode: str = "matrix",
                 filter_objective: bool = True,
                 constraint_filter_mask: None | np.ndarray = None,
                 el_flags: None | np.ndarray = None,
                 el_flags_policy: None | dict = None,
                 l: np.ndarray = np.array([1., 1.]),
                 **kwargs: Any) -> None:
        """
        Initialize filter and construct the neighborhood average.

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
            indicates how the neighborhood average is done. Possible values
            are "matrix", "helmholtz" or "convolution".
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
                                       el_flags_policy=el_flags_policy,
                                       kernel_fn=uniform_kernel)
        elif filter_mode == "helmholtz":
            self.filter = HelmholtzFilter(nelx=nelx,
                                          nely=nely,
                                          n_constr=n_constr,
                                          rmin=rmin,
                                          nelz=nelz,
                                          l=l)
        elif filter_mode == "convolution":
            self.filter = ConvolutionFilter(nelx=nelx,
                                            nely=nely,
                                            n_constr=n_constr,
                                            rmin=rmin,
                                            nelz=nelz,
                                            kernel_fn=uniform_kernel)
        else:
            raise ValueError("Unknown filter_mode: ", filter_mode)
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
                     beta: float,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x

        x_filtered = 1 + log( F(exp(beta*(x-1))) ) / beta

        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables, shape (n, k).
        beta : float
            filter strength.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables, shape (n, k).

        """
        return 1 + np.log(self.filter.apply_filter(x=np.exp(beta*(x-1))))/beta

    def apply_filter_dx(self,
                        x: np.ndarray,
                        dx_filtered: np.ndarray,
                        beta: float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables
        x_filtered using the chain rule assuming

            x_filtered = 1 + log( F(exp(beta*(x-1))) ) / beta

        to get the sensitivities with respect to the (unfiltered) design
        variables or in the case of many filters intermediate design variables:

            dx = exp(beta*(x-1)) * F^T( dx_filtered / F(exp(beta*(x-1))) )

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables, shape (n, k).
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables,
            shape (n, k).
        beta : float
            filter strength.

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables,
            shape (n, k).
        """
        ex = np.exp(beta*(x-1))
        return ex * self.filter.apply_filter_dx(x_filtered=None,
                                                dx_filtered=dx_filtered/self.filter.apply_filter(x=ex))

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

class ErodeFilterSigmund2007(TOFilter):
    """
    Implements the erode filter by

    Sigmund, Ole. "Morphology-based black and white filters for topology
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007):
    401-424.

    The erode filter is a smooth approximation of the minimum over the
    neighborhood of an element, so solid regions shrink. The filter equation
    is

    x_filtered = 1 - log( F(exp(beta*(1-x))) ) / beta

    where F is the normalized neighborhood average. In the original work the
    neighborhood is weighted uniformly within rmin which is used for
    filter_mode "matrix" and "convolution". For "helmholtz" the implicit
    kernel of the PDE filter replaces the uniform neighborhood. For numerical
    stability it is evaluated as

    x_filtered = - log( F(exp(-beta*x)) ) / beta.

    beta is the filter strength, that is typically ramped up during the TO
    process to large values. For beta to zero the filter reduces to the
    density filter, for beta to infinity to the minimum operator.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 rmin: float,
                 nelz: int | None = None,
                 filter_mode: str = "matrix",
                 filter_objective: bool = True,
                 constraint_filter_mask: None | np.ndarray = None,
                 el_flags: None | np.ndarray = None,
                 el_flags_policy: None | dict = None,
                 l: np.ndarray = np.array([1., 1.]),
                 **kwargs: Any) -> None:
        """
        Initialize filter and construct the neighborhood average.

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
            indicates how the neighborhood average is done. Possible values
            are "matrix", "helmholtz" or "convolution".
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
                                       el_flags_policy=el_flags_policy,
                                       kernel_fn=uniform_kernel)
        elif filter_mode == "helmholtz":
            self.filter = HelmholtzFilter(nelx=nelx,
                                          nely=nely,
                                          n_constr=n_constr,
                                          rmin=rmin,
                                          nelz=nelz,
                                          l=l)
        elif filter_mode == "convolution":
            self.filter = ConvolutionFilter(nelx=nelx,
                                            nely=nely,
                                            n_constr=n_constr,
                                            rmin=rmin,
                                            nelz=nelz,
                                            kernel_fn=uniform_kernel)
        else:
            raise ValueError("Unknown filter_mode: ", filter_mode)
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
                     beta: float,
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to (intermediate) design variables x

        x_filtered = - log( F(exp(-beta*x)) ) / beta

        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables, shape (n, k).
        beta : float
            filter strength.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables, shape (n, k).

        """
        return - np.log(self.filter.apply_filter(x=np.exp(-beta*x)))/beta

    def apply_filter_dx(self,
                        x: np.ndarray,
                        dx_filtered: np.ndarray,
                        beta: float,
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables
        x_filtered using the chain rule assuming

            x_filtered = - log( F(exp(-beta*x)) ) / beta

        to get the sensitivities with respect to the (unfiltered) design
        variables or in the case of many filters intermediate design variables:

            dx = exp(-beta*x) * F^T( dx_filtered / F(exp(-beta*x)) )

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables, shape (n, k).
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables,
            shape (n, k).
        beta : float
            filter strength.

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables,
            shape (n, k).
        """
        ex = np.exp(-beta*x)
        return ex * self.filter.apply_filter_dx(x_filtered=None,
                                                dx_filtered=dx_filtered/self.filter.apply_filter(x=ex))

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
