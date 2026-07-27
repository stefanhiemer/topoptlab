# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Tuple,Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.kernels import hat_kernel
from topoptlab.filter.matrix_filter import MatrixFilter
from topoptlab.filter.helmholtz_filter import HelmholtzFilter
from topoptlab.filter.convolution_filter import ConvolutionFilter

class SimplexFilter(TOFilter):
    """
    Filter that clips phase fractions to the unit simplex: sum_k x_k <= 1.

    Each row of x is divided by max(1, sum_k x_k), leaving rows already
    inside the simplex unchanged.
    """

    def __init__(self,
                 n_constr: int,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        n_constr : int
            Number of constraints.
        filter_objective : bool
            If True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            Boolean mask selecting which constraint sensitivities are filtered.
            None applies the filter to all constraints.
        """
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
                     **kwargs: Any) -> np.ndarray:
        """
        Project x onto the unit simplex row-wise.

        Parameters
        ----------
        x : np.ndarray, shape (n, k)
            Unfiltered design variables.

        Returns
        -------
        np.ndarray, shape (n, k)
        """
        return apply_simplex_projection(x)

    def apply_filter_dx(self,
                        x : np.ndarray,
                        dx_filtered : np.ndarray,
                        **kwargs: Any) -> np.ndarray:
        """
        Chain-rule pullback of sensitivities through the simplex projection.

        Parameters
        ----------
        x : np.ndarray, shape (n, k)
            Unfiltered design variables.
        dx_filtered : np.ndarray, shape (n, k)
            Sensitivities w.r.t. filtered variables.

        Returns
        -------
        np.ndarray, shape (n, k)
        """
        return apply_simplex_projection_dx(x, dx_filtered)

    @property
    def vol_conserv(self) -> bool:
        """False — this filter is not volume conserving."""
        return False

    @property
    def filter_objective(self) -> bool:
        """True if the filter is applied to objective sensitivities."""
        return self._filter_objective

    @property
    def constraint_filter_mask(self) -> np.ndarray:
        """Boolean array of shape (n_constr,) selecting filtered constraint sensitivities."""
        return self._constraint_filter_mask

    @property
    def changes_filter_kw(self) -> bool:
        return False

    def update_filter_kw(self, filter_kw: dict) -> None:
        return


def apply_simplex_projection(x: np.ndarray) -> np.ndarray:
    """
    Project x onto the unit simplex row-wise: x / max(1, sum_k x_k).

    Parameters
    ----------
    x : np.ndarray, shape (n, k)
        Unfiltered design variables.

    Returns
    -------
    x_filtered : np.ndarray, shape (n, k)
        Projected design variables.

    """
    return x / np.maximum(1., x.sum(axis=1, keepdims=True))


def apply_simplex_projection_dx(x: np.ndarray, dx_filtered: np.ndarray) -> np.ndarray:
    """
    Chain-rule pullback of sensitivities through apply_simplex_projection.

    Parameters
    ----------
    x : np.ndarray, shape (n, k)
        Unfiltered design variables.
    dx_filtered : np.ndarray, shape (n, k)
        Sensitivities w.r.t. filtered variables.

    Returns
    -------
    dx : np.ndarray, shape (n, k)
        Sensitivities w.r.t. unfiltered variables.

    """
    mat_frac = x.sum(axis=1, keepdims=True)
    active = (mat_frac > 1.)[:, 0]
    dx = dx_filtered.copy()
    dx[active] = dx_filtered[active] / mat_frac[active] - \
                 np.sum(dx_filtered[active] * x[active],
                        axis=1, keepdims=True) / mat_frac[active]**2
    return dx
