# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from scipy.sparse import csr_array

from topoptlab.filter.filter import TOFilter

# baseplate -> (axis, flip) with axis 0: x, 1: y, 2: z of the element index.
# Same convention as LangelaarFilter and design_analysis.baseplate_slice:
# N/S are the first/last image row (iy), W/E the first/last image column
# (ix) and B/U the first/last voxel layer (iz).
_BASEPLATES = {"W": (0, False),
               "E": (0, True),
               "N": (1, False),
               "S": (1, True),
               "B": (2, False),
               "U": (2, True)}


class CastingFilterGersborg2011(TOFilter):
    """
    Explicit casting parameterization by

    Gersborg, Allan Roulund, and Casper Schousboe Andreasen. "An explicit
    parameterization for casting constraints in gradient driven topology
    optimization." Structural and Multidisciplinary Optimization 44 (2011):
    875-881.

    One design variable per row of elements normal to the baseplate defines
    the position of the solid-void interface in that row. The element
    densities follow from a smooth Heaviside function and are monotonically
    decreasing with distance from the baseplate (one interface) so the
    design is castable by construction. With two interfaces, a second,
    flipped Heaviside function adds a solid part at the face opposite to the
    baseplate.

    For element i in row j with normalized distance s_i in (0,1) of its
    center from the baseplate, the density is (Eq. 11)

        rho_hat_j = (a+b) rho_j - b
        x_i       = H(s_i, rho_hat_j; beta)

    which is evaluated equivalently by stretching the coordinate instead of
    the interface position,

        s'_i = clip((s_i + b)/(a + b), 0, 1)
        x_i  = H(s'_i, rho_j; beta).

    H is the smooth, decreasing Heaviside function of Eq. 10

        s <= rho: H = 1 - rho [exp(-beta (1 - s/rho)) - (1 - s/rho) exp(-beta)]
        s >  rho: H = 1 - (1-rho) [1 - exp(-beta (s-rho)/(1-rho))
                                   + (s-rho) exp(-beta)/(1-rho)] - rho

    with H(0) = 1, H(rho) = 1 - rho and H(1) = 0. For two interfaces with
    design variables rho = [rho_1, rho_2] (Eq. 12)

        x_1 = H(s', rho_1; beta)
        x_2 = 1 - H(s', rho_2; beta)
        x   = x_1 + (1 - x_1) x_2.

    The design variables are ordered like the elements of the baseplate
    face, i.e. y fastest, then x, then z, consistent with map_eltoimg /
    map_eltovoxel. For baseplate 'B'/'U' the design vector is thus an
    ordinary 2D element field of shape (nelx*nely,).

    The filter maps design variables of shape (n_design, k) to element
    densities of shape (n_el, k), i.e. it changes the number of variables and
    must be the first filter in a chain.
    """

    def __init__(self,
                 nelx: int,
                 nely: int,
                 n_constr: int,
                 baseplate: str = "W",
                 a: float = 1.,
                 b: float = 0.,
                 n_interfaces: int = 1,
                 filter_objective: bool = True,
                 constraint_filter_mask: None | np.ndarray = None,
                 nelz: None | int = None,
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
        baseplate : str
            face at which the material is attached, same convention as
            LangelaarFilter and design_analysis.support_indicator. In 2D:
            'N', 'S', 'E', 'W' (first/last image row, first/last image
            column). In 3D additionally 'B', 'U' (first/last voxel layer in
            z). Material is solid at the baseplate and void away from it.
        a : float
            upper limit of the stretched interface position. a > 1 allows
            solid at the face opposite to the baseplate.
        b : float
            negative lower limit of the stretched interface position, i.e.
            the interface position lies in [-b, a]. b > 0 allows void at the
            baseplate, b < 0 enforces a solid wall there.
        n_interfaces : int
            1 (Eq. 11) or 2 (Eq. 12). For 2, the design variables are
            [rho_1, rho_2] stacked and n_design = 2 * number of rows.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
        nelz : int or None
            number of elements in z direction. None for 2D.
        """
        #
        baseplate = baseplate.upper()
        if baseplate not in _BASEPLATES:
            raise ValueError("baseplate must be one of 'N','S','E','W','B','U', "
                             f"got {baseplate}.")
        if nelz is None and baseplate in "BU":
            raise ValueError(f"baseplate {baseplate} requires nelz (3D).")
        if n_interfaces not in (1, 2):
            raise ValueError(f"n_interfaces must be 1 or 2, got {n_interfaces}.")
        if not a + b > 0:
            raise ValueError(f"a + b must be positive, got a={a}, b={b}.")
        #
        self.baseplate = baseplate
        self.a = a
        self.b = b
        self.n_interfaces = n_interfaces
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
        # element indices per axis following the element ordering of
        # map_eltoimg / map_eltovoxel
        if nelz is None:
            shape = (nelx, nely)
        else:
            shape = (nelx, nely, nelz)
        n_el = int(np.prod(shape))
        el = np.arange(n_el)
        inds = [(el // nely) % nelx, el % nely]
        if nelz is not None:
            inds.append(el // (nelx * nely))
        # normalized distance of element centers from the baseplate
        axis, flip = _BASEPLATES[baseplate]
        s = (inds[axis] + 0.5) / shape[axis]
        if flip:
            s = 1. - s
        # stretch: evaluating the Heaviside at s' with the interface position
        # rho in [0,1] is the same as placing the interface at
        # rho_hat = (a+b)*rho - b in the original coordinate s. Clipping
        # makes elements outside the stretched interval fully solid/void.
        self.s = np.clip((s + b) / (a + b), 0., 1.)
        # index of the row (design variable) each element belongs to. The
        # remaining axes are ordered from slowest to fastest as in the
        # element numbering (z, x, y) so the design variables follow the
        # map_eltoimg / map_eltovoxel conventions.
        other = [i for i in (2, 0, 1) if i != axis and i < len(shape)]
        self.n_rows = int(np.prod([shape[i] for i in other]))
        self.row_idx = np.ravel_multi_index([inds[i] for i in other],
                                            [shape[i] for i in other])
        self.n_el = n_el
        # row selection matrix of shape (n_rows, n_el) that sums element
        # contributions per row, i.e. the transpose of x[row_idx]
        self.P = csr_array((np.ones(n_el), (self.row_idx, el)),
                           shape=(self.n_rows, n_el))
        return

    @property
    def n_design(self) -> int:
        """
        Number of design variables expected as input of the filter.
        """
        return self.n_interfaces * self.n_rows

    def apply_filter(self,
                     x: np.ndarray,
                     beta: float,
                     **kwargs: Any) -> np.ndarray:
        """
        Map design variables (interface positions) to element densities.

        Parameters
        ----------
        x : np.ndarray
            design variables in [0,1], shape (n_design,) or (n_design, k).
        beta : float
            steepness of the smooth Heaviside function.

        Returns
        -------
        x_filtered : np.ndarray
            element densities, shape (n_el,) or (n_el, k).
        """
        if x.shape[0] != self.n_design:
            raise ValueError(f"expected {self.n_design} design variables, "
                             f"got {x.shape[0]}.")
        if x.ndim == 1:
            s = self.s
        else:
            s = self.s[:, None]
        # first interface: solid at start of row
        h = smooth_heaviside(s=s,
                             rho=x[:self.n_rows][self.row_idx],
                             beta=beta)
        if self.n_interfaces == 2:
            # second, flipped interface: solid at end of row
            h2 = 1. - smooth_heaviside(s=s,
                                       rho=x[self.n_rows:][self.row_idx],
                                       beta=beta)
            h = h + (1. - h) * h2
        return h

    def apply_filter_dx(self,
                        x: np.ndarray,
                        dx_filtered: np.ndarray,
                        beta: float,
                        **kwargs: Any) -> np.ndarray:
        """
        Chain-rule pullback of sensitivities through the casting filter.
        Each element depends only on the design variable(s) of its row, so

            dx_j = sum_{i in row j} dx_filtered_i * d x_filtered_i / d rho_j

        with d x_filtered_i / d rho_j = dH/drho (s'_i, rho_j) for one
        interface and for two interfaces (x = x_1 + (1 - x_1) x_2)

            d x / d rho_1 = (1 - x_2) dH/drho (s', rho_1)
            d x / d rho_2 = -(1 - x_1) dH/drho (s', rho_2).

        Parameters
        ----------
        x : np.ndarray
            design variables, shape (n_design,) or (n_design, k).
        dx_filtered : np.ndarray
            sensitivities w.r.t. element densities, shape (n_el,) or
            (n_el, m) with m = k or k = 1 (e.g. several constraints).
        beta : float
            steepness of the smooth Heaviside function.

        Returns
        -------
        dx : np.ndarray
            sensitivities w.r.t. design variables, shape (n_design,) or
            (n_design, m).
        """
        if x.shape[0] != self.n_design:
            raise ValueError(f"expected {self.n_design} design variables, "
                             f"got {x.shape[0]}.")
        if x.ndim == 1:
            s = self.s
        else:
            s = self.s[:, None]
        # element-wise derivatives / values of the Heaviside functions
        rho1 = x[:self.n_rows][self.row_idx]
        dh1 = smooth_heaviside_drho(s=s, rho=rho1, beta=beta)
        if self.n_interfaces == 2:
            rho2 = x[self.n_rows:][self.row_idx]
            h1 = smooth_heaviside(s=s, rho=rho1, beta=beta)
            h2 = 1. - smooth_heaviside(s=s, rho=rho2, beta=beta)
            dh2 = -(1. - h1) * smooth_heaviside_drho(s=s, rho=rho2, beta=beta)
            dh1 = (1. - h2) * dh1
            derivs = [dh1, dh2]
        else:
            derivs = [dh1]
        # broadcast element derivatives against several sensitivity columns
        if dx_filtered.ndim == 2 and x.ndim == 1:
            derivs = [d[:, None] for d in derivs]
        return np.concatenate([self.P @ (dx_filtered * d) for d in derivs],
                              axis=0)

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


def smooth_heaviside(s: np.ndarray,
                     rho: np.ndarray,
                     beta: float) -> np.ndarray:
    """
    Smooth, decreasing Heaviside function of Eq. (10) in Gersborg and
    Andreasen (2011), i.e. one minus the threshold projection of

    Xu, Shengli, Yuanwu Cai, and Gengdong Cheng. "Volume preserving nonlinear
    density filter based on heaviside functions." Structural and
    Multidisciplinary Optimization 41 (2010): 495-505.

        s <= rho: H = 1 - rho [exp(-beta (1 - s/rho)) - (1 - s/rho) exp(-beta)]
        s >  rho: H = 1 - (1-rho) [1 - exp(-beta (s-rho)/(1-rho))
                                   + (s-rho) exp(-beta)/(1-rho)] - rho

    H(0) = 1, H(rho) = 1 - rho, H(1) = 0 and H approaches 1 for s < rho and 0
    for s > rho as beta goes to infinity.

    Parameters
    ----------
    s : np.ndarray
        normalized coordinate in [0,1].
    rho : np.ndarray
        interface position in [0,1], broadcastable to s.
    beta : float
        steepness.

    Returns
    -------
    H : np.ndarray
        smooth Heaviside function values in [0,1].
    """
    eps = np.finfo(float).tiny
    # guard the divisions; the guarded branch is never selected where the
    # denominator vanishes (rho=0 implies s>=rho, rho=1 implies s<=rho)
    r = np.maximum(rho, eps)
    rc = np.maximum(1. - rho, eps)
    #
    t = 1. - s / r
    h_left = 1. - r * (np.exp(-beta * np.maximum(t, 0.)) - t * np.exp(-beta))
    #
    d = s - rho
    h_right = 1. - rc * (1. - np.exp(-beta * np.maximum(d, 0.) / rc)
                         + d * np.exp(-beta) / rc) - rho
    return np.where(s <= rho, h_left, h_right)

def smooth_heaviside_drho(s: np.ndarray,
                          rho: np.ndarray,
                          beta: float) -> np.ndarray:
    """
    Derivative of the smooth Heaviside function H(s, rho; beta) of
    smooth_heaviside with respect to the interface position rho.

        s <= rho: dH/drho = exp(-beta) - E (1 - beta s/rho),
                  E = exp(-beta (1 - s/rho))
        s >  rho: dH/drho = exp(-beta) - F (1 - beta (1-s)/(1-rho)),
                  F = exp(-beta (s-rho)/(1-rho))

    The second branch follows from simplifying H to
    H = (1-rho) F - (s-rho) exp(-beta). Both branches give
    exp(-beta) - (1 - beta) at s = rho, i.e. dH/drho is continuous.

    Parameters
    ----------
    s : np.ndarray
        normalized coordinate in [0,1].
    rho : np.ndarray
        interface position in [0,1], broadcastable to s.
    beta : float
        steepness.

    Returns
    -------
    dH : np.ndarray
        derivative of H with respect to rho.
    """
    eps = np.finfo(float).tiny
    # same guards as in smooth_heaviside
    r = np.maximum(rho, eps)
    rc = np.maximum(1. - rho, eps)
    #
    t = 1. - s / r
    dh_left = np.exp(-beta) - np.exp(-beta * np.maximum(t, 0.)) \
              * (1. - beta * s / r)
    #
    d = s - rho
    dh_right = np.exp(-beta) - np.exp(-beta * np.maximum(d, 0.) / rc) \
               * (1. - beta * (1. - s) / rc)
    return np.where(s <= rho, dh_left, dh_right)
