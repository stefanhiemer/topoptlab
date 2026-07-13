# SPDX-License-Identifier: GPL-3.0-or-later
from typing import  Union

import numpy as np

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.fetch_filter import fetch_filters

def sensitivity_propagation(dobj: np.ndarray,
                            dconstrs: np.ndarray,
                            x: np.ndarray,
                            xTilde: List[np.ndarray],
                            xPhys: np.ndarray,
                            ft: list,
                            filter_kw: Union[Dict,List],
                            el_flags_policy: Union[None,Dict],
                            prescribed_mask: np.ndarray,
                            **kwargs: Any
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate objective and constraint sensitivities backward through the filter
    chain via the chain rule, mapping gradients from physical variables back
    to design variables.

    Given a filter chain ``xPhys = ft[-1](ft[-2](...ft[0](x)...))``, the
    backward pass applies each filter's adjoint in reverse order:

        dx = ft[0].T( ft[1].T( ... ft[-1].T(dxPhys) ... ) )

    Each filter's ``apply_filter_dx`` is called only when its
    ``filter_objective`` / ``constraint_filter_mask`` flags allow it.
    After the filter chain, sensitivities at prescribed (passive/active)
    elements are zeroed when ``el_flags_policy["correct_backward"]`` is True.

    Parameters
    ----------
    dobj : np.ndarray, shape (n, k)
        Sensitivities of the objective w.r.t. physical densities. Modified
        in-place and returned.
    dconstrs : np.ndarray, shape (n, n_constr)
        Sensitivities of the constraints w.r.t. physical densities. Modified
        in-place and returned.
    x : np.ndarray, shape (n, k)
        Current design variables (unfiltered).
    xTilde : list of np.ndarray, each shape (n, k)
        Intermediate filtered variables between consecutive filter stages.
        ``xTilde[i]`` is the output of ``ft[i]``. Empty when ``len(ft) == 1``.
    xPhys : np.ndarray, shape (n, k)
        Physical densities, i.e. the output of the last filter ``ft[-1]``.
    ft : list of TOFilter
        Ordered filter chain. Must be a non-empty list.
    filter_kw : dict
        Keyword arguments forwarded to each filter's ``apply_filter_dx``.
    el_flags_policy : dict or None
        Policy dict controlling element flag enforcement. If not None and
        ``el_flags_policy["correct_backward"]`` is True, sensitivities at
        prescribed elements are set to zero after filtering.
    prescribed_mask : np.ndarray of bool, shape (n,) or None
        Boolean mask identifying prescribed (passive or active) elements.
        Only used when ``el_flags_policy["correct_backward"]`` is True.

    Returns
    -------
    dobj : np.ndarray, shape (n, k)
        Objective sensitivities w.r.t. design variables.
    dconstrs : np.ndarray, shape (n, n_constr)
        Constraint sensitivities w.r.t. design variables.
    """

    if isinstance(ft, list):
        #
        if ft[-1].filter_objective:
            dobj[:] = ft[-1].apply_filter_dx(x=x if len(ft)==1 else xTilde[-1],
                                             x_filtered=xPhys,
                                             dx_filtered=dobj,
                                             **filter_kw)
        #
        if np.any(ft[-1].constraint_filter_mask):
            dconstrs[:,ft[-1].constraint_filter_mask] = \
                 ft[-1].apply_filter_dx(x=x if len(ft)==1 else xTilde[-1],
                                        x_filtered=xPhys,
                                        dx_filtered=dconstrs[:,ft[-1].constraint_filter_mask],
                                        **filter_kw)
        if len(ft) > 1:
            for i in range(len(ft)-2,-1,-1):
                #
                if ft[i].filter_objective:
                    dobj[:] = ft[i].apply_filter_dx(x=x if i==0 else xTilde[i-1],
                                                    x_filtered=xTilde[i],
                                                    dx_filtered=dobj,
                                                    **filter_kw)
                #
                if np.any(ft[i].constraint_filter_mask):
                    dconstrs[:,ft[i].constraint_filter_mask] =\
                        ft[i].apply_filter_dx(x=x if i==0 else xTilde[i-1],
                                              x_filtered=xTilde[i],
                                              dx_filtered=dconstrs[:,ft[i].constraint_filter_mask],
                                              **filter_kw)
    else:
        raise ValueError(f"No filter applied. ft: {ft}")
    # backward filter policy: zero sensitivities at prescribed elements
    if el_flags_policy is not None and el_flags_policy["correct_backward"]:
        dobj[prescribed_mask] = 0.
        dconstrs[prescribed_mask] = 0.

    return dobj, dconstrs

def filter_design_variables(x: np.ndarray,
                            xTilde: List[np.ndarray],
                            xPhys: np.ndarray,
                            ft: list,
                            filter_kw: Union[Dict,List],
                            el_flags_policy: Union[None,Dict],
                            passive_mask: np.ndarray,
                            active_mask: np.ndarray,
                            **kwargs: Any) -> np.ndarray:
    """
    Apply the filter chain in the forward direction, mapping design variables
    to physical densities.

    Given a filter chain ``ft = [ft[0], ft[1], ..., ft[-1]]``, computes:

        xTilde[0] = ft[0](x)
        xTilde[i] = ft[i](xTilde[i-1])   for i = 1, ..., len(ft)-2
        xPhys     = ft[-1](xTilde[-1])

    For a single filter (``len(ft) == 1``) the intermediate list is unused and
    ``xPhys = ft[0](x)`` directly. Filters that carry internal state (e.g. an
    adaptive eta projector) update ``filter_kw`` in-place via
    ``update_filter_kw`` after each stage. After the chain, prescribed element
    densities are restored to 0 (passive) or 1 (active) when
    ``el_flags_policy["correct_forward"]`` is True.

    Parameters
    ----------
    x : np.ndarray, shape (n, k)
        Current design variables (unfiltered).
    xTilde : list of np.ndarray, each shape (n, k)
        Intermediate filtered variables. ``xTilde[i]`` receives the output of
        ``ft[i]`` and is modified in-place. Length must be ``len(ft) - 1``.
    xPhys : np.ndarray, shape (n, k)
        Physical densities to be updated with the output of the last filter.
    ft : list of TOFilter
        Ordered filter chain. Must be a non-empty list.
    filter_kw : dict
        Keyword arguments forwarded to each filter's ``apply_filter`` and
        updated in-place by any filter whose ``changes_filter_kw`` is True.
    el_flags_policy : dict or None
        Policy dict controlling element flag enforcement. If not None and
        ``el_flags_policy["correct_forward"]`` is True, prescribed densities
        are restored after filtering.
    passive_mask : np.ndarray of bool, shape (n,) or None
        Boolean mask identifying passive elements (density forced to 0).
    active_mask : np.ndarray of bool, shape (n,) or None
        Boolean mask identifying active elements (density forced to 1).

    Returns
    -------
    xTilde : list of np.ndarray
        Updated intermediate filtered variables.
    xPhys : np.ndarray, shape (n, k)
        Updated physical densities.
    """
    if isinstance(ft, list):
        if len(ft) > 1:
            xTilde[0] = ft[0].apply_filter(x=x,
                                           **filter_kw)
            for i in range(1,len(ft)-1):
                xTilde[i] = ft[i].apply_filter(x=xTilde[i-1],
                                               **filter_kw)
                #
                if ft[i].changes_filter_kw:
                    ft[i].update_filter_kw(filter_kw)
            xPhys = ft[-1].apply_filter(x=xTilde[-1],
                                        **filter_kw)
            #
            if ft[-1].changes_filter_kw:
                ft[-1].update_filter_kw(filter_kw)
        else:
            xPhys = ft[0].apply_filter(x=x,
                                        **filter_kw)
            #
            if ft[0].changes_filter_kw:
                ft[0].update_filter_kw(filter_kw)
        for _f in ft:
            if _f.changes_filter_kw:
                _f.update_filter_kw(filter_kw)
    else:
        raise TypeError(f"ft should be a list at this point: {type(ft)}")
    # forward filter policy: restore prescribed densities after filtering
    if el_flags_policy is not None and el_flags_policy["correct_forward"]:
        xPhys[passive_mask] = 0.
        xPhys[active_mask]  = 1.
    return xTilde, xPhys

def prepare_filters(ft: Union[int, type, List],
                    nelx: int,
                    nely: int,
                    nelz: Union[int, None],
                    filter_mode: str,
                    rmin: float,
                    n_constr: int,
                    l: Union[float, np.ndarray],
                    el_flags: Union[None, np.ndarray],
                    el_flags_policy: Union[None, Dict],
                    filter_kw: Dict,
                    mapping: Callable,
                    invmapping: Callable,
                    constraint_filter_mask: np.ndarray,
                    **kwargs: Any) -> List[TOFilter]:
    """
    Instantiate the filter chain from the ``ft`` specification and return it
    as an ordered list of :class:`TOFilter` objects.

    Three calling conventions are accepted for ``ft``:

    * **TOFilter subclass** (not int, not list): a single filter is constructed
      by calling ``ft(nelx=..., rmin=..., ...)``.
    * **list of TOFilter subclasses**: each entry is instantiated with the same
      mesh and filter keyword arguments; the resulting list defines the chain in
      forward order.
    * **int**: the integer code is resolved to a standard filter chain via
      :func:`fetch_filters`.

    Parameters
    ----------
    ft : int, TOFilter subclass, or list of TOFilter subclasses
        Filter specification.
    nelx : int
        Number of elements in x-direction.
    nely : int
        Number of elements in y-direction.
    nelz : int or None
        Number of elements in z-direction; ``None`` for 2-D problems.
    filter_mode : str
        Filter implementation mode, e.g. ``"convolution"`` or ``"matrix"``.
    rmin : float
        Filter cut-off radius.
    n_constr : int
        Number of constraints; passed to the filter to size the constraint
        sensitivity mask.
    l : float or np.ndarray
        Element size. Broadcast to all spatial directions when scalar.
    el_flags : np.ndarray or None
        Integer element-flag array (0 free, 1 passive, 2 active).
    el_flags_policy : dict or None
        Policy controlling how element flags are enforced in the filter.
    filter_kw : dict
        Additional keyword arguments forwarded to every filter constructor.
    mapping : callable
        Maps a flat element array to image/voxel layout.
    invmapping : callable
        Maps image/voxel layout back to a flat element array.
    constraint_filter_mask : np.ndarray of bool, shape (n_constr,)
        Indicates which constraint sensitivities are filtered; passed to the
        filter when constructing from an integer code.

    Returns
    -------
    ft : list of TOFilter
        Instantiated filter chain in forward application order.
    """
    if not isinstance(ft, (int, list)) and issubclass(ft, TOFilter):
        ft = [ft(nelx=nelx, nely=nely, nelz=nelz,
                 filter_mode=filter_mode,
                 rmin=rmin,
                 n_constr=n_constr,
                 l=l,
                 el_flags=el_flags,
                 el_flags_policy=el_flags_policy,
                 **filter_kw)]
    elif isinstance(ft, list):
        ft = [ft_obj(nelx=nelx, nely=nely, nelz=nelz,
                     filter_mode=filter_mode,
                     rmin=rmin,
                     n_constr=n_constr,
                     l=l,
                     el_flags=el_flags,
                     el_flags_policy=el_flags_policy,
                     **filter_kw)
              for ft_obj in ft]
    elif isinstance(ft, int):
        _ft_kw = dict(nelx=nelx, nely=nely, nelz=nelz,
                      filter_mode=filter_mode,
                      rmin=rmin,
                      n_constr=n_constr,
                      l=l,
                      mapping=mapping,
                      invmapping=invmapping,
                      constraint_filter_mask=constraint_filter_mask,
                      el_flags=el_flags,
                      el_flags_policy=el_flags_policy,
                      **filter_kw)
        ft = fetch_filters(ft=ft,
                           filter_args=[_ft_kw])
    else:
        raise ValueError(f"Unknown filter. ft: {ft} filter_mode {filter_mode}")
    return ft
