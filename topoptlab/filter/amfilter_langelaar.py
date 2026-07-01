# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial
from typing import Any, Callable, Union

import numpy as np
from scipy.sparse import csc_array

from topoptlab.filter.filter import TOFilter
from topoptlab.utils import map_eltoimg, map_imgtoel


class LangelaarFilter(TOFilter):
    """
    Additive manufacturing filter by

    Langelaar, Matthijs. "An additive manufacturing filter for topology
    optimization of print-ready designs." Structural and multidisciplinary
    optimization 55 (2017): 871-883.

    Wraps the AMfilter function as a TOFilter. Only 2D is supported.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 mapping: Callable, 
                 invmapping: Callable,
                 baseplate: str = 'S',
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 nelz: Union[None, int] = None,
                 channel: Union[None,int] = 0,
                 **kwargs: Any) -> None:
        """
        Parameters
        ----------
        nelx : int
            number of elements in x direction.
        nely : int
            number of elements in y direction.
        n_constr : int
            number of constraints.
        mapping : callable
            maps 1D design variable array of shape (n, k) to a 3D array of
            shape (nely, nelx, k) suitable for the AM filter.
        invmapping : callable
            inverse of mapping; maps (nely, nelx, k) back to (n, k).
        baseplate : str
            baseplate orientation: 'N', 'E', 'S' (default), 'W', or 'X'
            (bypass).
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
        nelz : int or None
            must be None; 3D is not supported.
        channel : None or int
            channel to which to apply filter. If None all channels are summed over 
            and filter is applied to result. Assume that x.sum(axis) \leq 1 where 
            1-x is void.
        """
        #
        if nelz is not None:
            raise NotImplementedError("AMFilter is only implemented for 2D.")
        #
        self.baseplate = baseplate
        if channel is None:
            self.channel = -1
        else:
            self.channel = channel
        self.mapping = mapping
        self.invmapping = invmapping
        self._filter_objective = filter_objective
        #
        if constraint_filter_mask is None:
            self._constraint_filter_mask = np.ones(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
                constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        else:
            raise TypeError(
                "constraint_filter_mask must be None or np.ndarray of shape (n_constr,).")

    def apply_filter(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Apply the AM filter to design variables x.

        Parameters
        ----------
        x : np.ndarray
            design variables, shape (n,) or (n, k).

        Returns
        -------
        x_filtered : np.ndarray
            printable design densities, same shape as x.
        """
        # multi-channel: x_img shape (nely, nelx, k)
        if self.channel == -1:
            return self.invmapping(AMfilter(self.mapping(x).sum(axis=-1),
                                            baseplate=self.baseplate)[:, :, None])
        else:
            return self.invmapping(AMfilter(self.mapping(x)[..., self.channel],
                                            baseplate=self.baseplate)[:, :, None])

    def apply_filter_dx(self,
                        x: np.ndarray,
                        x_filtered: np.ndarray,
                        dx_filtered: np.ndarray,
                        **kwargs: Any) -> np.ndarray:
        """
        Chain-rule pullback of sensitivities through the AM filter.

        Parameters
        ----------
        x : np.ndarray
            unfiltered design variables, shape (n,) or (n, k).
        x_filtered : np.ndarray
            filtered design variables from the forward pass, same shape as x.
            Used in sum mode to recover xi_solid without rerunning the forward.
        dx_filtered : np.ndarray
            sensitivities w.r.t. filtered variables, same shape as x.

        Returns
        -------
        dx : np.ndarray
            sensitivities w.r.t. x, same shape as x.
        """
        if self.channel == -1:
            dx_new = AMfilter(self.mapping(x).sum(axis=-1),
                              baseplate=self.baseplate,
                              sensitivities=self.mapping(dx_filtered)) \
                     * np.ones((1, 1, x.shape[-1]))
        else:
            # channel mode: only self.channel contributes, others get zero
            dx_single = AMfilter(self.mapping(x)[:, :, self.channel],
                                 baseplate=self.baseplate,
                                 sensitivities=self.mapping(dx_filtered))[:, :, 0]  # (nely, nelx)
            dx_new = np.zeros(dx_single.shape + (x.shape[-1],))               # (nely, nelx, k)
            dx_new[:, :, self.channel] = dx_single
        return self.invmapping(dx_new)

    @property
    def vol_conserv(self) -> bool:
        return False

    @property
    def filter_objective(self) -> bool:
        return self._filter_objective

    @property
    def constraint_filter_mask(self) -> np.ndarray:
        return self._constraint_filter_mask

    @property
    def changes_filter_kw(self) -> bool:
        return False

    def update_filter_kw(self, filter_kw: dict) -> None:
        return

def AMfilter(x: np.ndarray, 
             baseplate: str = 'S',
             sensitivities: Union[np.ndarray,None] = None) -> np.ndarray:
    """
    Applies the filter by

    Langelaar, Matthijs. "An additive manufacturing filter for topology
    optimization of print-ready designs." Structural and multidisciplinary
    optimization 55 (2017): 871-883.

    Applies a filter to densities that enforces that each density cannot be
    larger then the maximum density of its supporting region.

    Parameters
    ----------
    x : np.ndarray
        Blueprint design (2D array), with values between 0 and 1 and shape
        (nely,nelx). The shape is needed to determine the positions of elements
        with respect to the baseplate.
    baseplate : str, optional
        Character indicating baseplate orientation: 'N', 'E', 'S', 'W'. Default is 'S'.
        For 'X', the filter bypasses and returns the input as-is.
    sensitivities : np.ndarray
        sensitivities associated with the design input shape
        (nely,nelx,nsens).

    Returns
    -------
    xi or sensitivities: np.ndarray
        Printable design density (nely,nelx) or sensitivities (nely,nelx,nSens)
        after filtering.
    """
    # number of supporting elements
    Ns = 3
    # constants for smooth max/min functions
    P,ep,xi_0 = 40,1e-4,.5
    Q = P + np.log(Ns) / np.log(xi_0)
    SHIFT = 100 * np.finfo(float).tiny**(1 / P)
    BACKSHIFT = 0.95 * Ns**(1 / Q) * SHIFT**(P / Q)
    # check for bypass option
    if baseplate == 'X':
        return x, sensitivities  # Return as-is
    # determine rotation based on baseplate orientation
    nRot = 'SWNE'.find(baseplate.upper())
    x = np.rot90(x, nRot).copy()
    # initialize xi
    xi = np.zeros_like(x)
    nely, nelx = x.shape
    # loop for applying AM filter from top moving layer-wise downwards
    xi[-1, :] = x[-1,:].copy()
    Xi, keep,sq = [np.zeros_like(x) for i in np.arange(3)]
    for i in np.arange(nely-2,-1,-1):
        cbr = np.pad(xi[i+1,:] + SHIFT,
                     (1, 1),
                     'constant',
                     constant_values=SHIFT)
        keep[i,:] = (cbr[:-2]**P + cbr[1:-1]**P + cbr[2:]**P)
        Xi[i,:] = keep[i,:]**(1 / Q) - BACKSHIFT
        sq[i,:] = np.sqrt((x[i,:] - Xi[i,:])**2 + ep)
        xi[i,:] = 0.5 * ((x[i,:] + Xi[i,:]) - sq[i,:] + np.sqrt(ep))
    # process sensitivities if provided.
    if sensitivities is not None:
        #
        nSens = sensitivities.shape[-1]
        # sensitivities as obtained by the usual adjoint analysis. this must be
        # rotated and filtered
        dfxi = np.rot90(sensitivities, nRot)
        # filtered gradients/sensitivities
        dfx = np.zeros_like(dfxi)
        # precalculate indices later for fast multiplication via sparse matrix
        qi = np.repeat(np.arange(nelx), Ns)
        qj = np.tile([-1, 0, 1], nelx) + qi
        # Lagrangian multipliers for adjoint sensitivity analysis
        lambda_vals = np.zeros((nelx,nSens))
        # iterate from top to base layer
        for i in np.arange(nely-1):
            # smin sensitivity terms
            dsmindx = 0.5 * (1 - (x[i, :] - Xi[i, :]) / sq[i, :])
            dsmindXi = 1 - dsmindx
            # smax sensitivity terms
            cbr = np.pad(xi[i + 1, :] + SHIFT,
                         (1, 1),
                         'constant',
                         constant_values=SHIFT)  # Pad with zeros
            dmx = np.zeros((nelx,Ns))
            for j in np.arange(Ns):
                dmx[:,j] = (P/Q) * keep[i, :]**((1/Q) - 1) * cbr[np.arange(nelx)+j]**(P - 1)
            # rearrange data for quick multiplication
            qs = dmx.flatten()
            dsmaxdxi = csc_array((qs[1:-1],(qi[1:-1], qj[1:-1])),
                                   shape=(nelx, nelx))
            # update sensitivities
            for k in np.arange(nSens):
                dfx[i,:,k] = dsmindx * (dfxi[i,:,k] + lambda_vals[:,k])
                lambda_vals[:,k] = ((dfxi[i,:,k] + lambda_vals[:,k]) * dsmindXi) @ dsmaxdxi
        # base layer
        dfx[-1,:,:] = dfxi[-1,:,:]+lambda_vals[:,:]
    if sensitivities is None:
        # rotate xi back to original orientation if rotated
        return np.rot90(xi, -nRot)
    else:
        # rotate sensitivities back to original orientation if rotated
        return np.rot90(dfx, -nRot)