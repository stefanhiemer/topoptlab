# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Dict, List, Union

import numpy as np

from topoptlab.log_utils import EmptyLogger,SimpleLogger
from topoptlab.design_analysis import gray_indicator, level_indicator

def run_continuation(continuation_kw: Dict,
                     stage: int,
                     **call_args: Any) -> bool:
    """
    Run all continuation functions registered for ``stage`` and return the
    global stop flag which is used to determine when to stop the optimization.

    Each entry in ``continuation_kw["func_kws"]`` may carry a ``"stage"``
    key (default 1).  Only functions whose stage matches the ``stage``
    argument are called; the others are skipped but their last-recorded
    stop flag is preserved.

    After calling the matching functions, the function returns
    ``all(continuation_kw["stop_flag"])``, i.e. True only when every
    registered function (across all stages) has signalled that it is done.

    Parameters
    ----------
    continuation_kw : dict
        continuation keyword dictionary. Must contain:

        - ``"funcs"``     : list of callables.
        - ``"func_kws"``  : list of per-function dicts, each optionally
          carrying ``"stage"`` (default 1) and any kwargs forwarded to the
          function via ``**``.
        - ``"stop_flag"`` : list of bool, one per function, updated in-place.
    stage : int
        which hook point to execute (0 = before optimizer, 1 = after optimizer).
    **call_args :
        keyword arguments forwarded to every called function in addition to
        the function's own ``func_kw`` entries. Extra keys not accepted by a
        function are silently absorbed by its ``**kwargs``.

    Returns
    -------
    stop : bool
        True if every function across all stages has returned True.
    """
    flags = continuation_kw["stop_flag"]
    for i, (func, kw) in enumerate(zip(continuation_kw["funcs"],
                                       continuation_kw["func_kws"])):
        if kw.get("stage", 1) == stage:
            flags[i] = func(**call_args, **kw)
    return all(flags)

def update_move_limit(filter_kw: Dict,
                      optimizer_kw: Dict,
                      transition_eps: float = 0.01,
                      varsigma: float = 5.0,
                      move_min: float = 0.01,
                      move_max: float = 0.2,
                      n_sample: int = 1000,
                      logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                      **kwargs: Any) -> None:
    """
    Adapt the move limit to the current width of the Heaviside transition zone.

    Estimates the width ``z`` of the transition interval where the smooth
    Heaviside projection (tanh form, Xu 2010) is strictly between
    ``transition_eps`` and ``1 - transition_eps``, then sets

        move = clip(z / varsigma, move_min, move_max)

    and writes the result in-place to ``optimizer_kw["move"]``.

    Parameters
    ----------
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``; ``"eta"`` is
        read if present (default 0.5).
    optimizer_kw : dict
        MMA keyword dictionary. ``"move"`` is updated in-place.
    transition_eps : float
        projection value used to define the transition zone boundaries;
        the active interval is where ``transition_eps < H < 1 - transition_eps``
        (default 0.01).
    varsigma : float
        scaling divisor converting the zone width to the move limit
        (default 5.0).
    move_min : float
        lower clip bound on the move limit (default 0.01).
    move_max : float
        upper clip bound on the move limit (default 0.2).
    n_sample : int
        number of sample points in [0, 1] used to estimate the transition
        width (default 1000).
    logger : BaseLogger
        logger object.
    
    Returns
    -------
    True : bool
        this function does not give any criteria about converging, so it is
        always ready to stop.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    if "eta" not in filter_kw:
        raise ValueError('"eta" is not in filter_kw: ', filter_kw)
    beta = filter_kw["beta"]
    eta = filter_kw["eta"]
    xs = np.linspace(0.0, 1.0, n_sample)
    #
    denom = np.tanh(beta * eta) + np.tanh(beta * (1. - eta))
    if denom < 1e-16:
        denom = 1e-16
    ys     = (np.tanh(beta * eta) + np.tanh(beta * (xs - eta))) / denom
    active = (ys > transition_eps) & (ys < 1.0 - transition_eps)
    if active.any():
        z = xs[active].max() - xs[active].min()
    else:
        z = 1.0 / beta if beta > 0.0 else move_min
    move = float(np.clip(z / varsigma, move_min, move_max))
    optimizer_kw["move"] = move
    logger.info("move limit adapted to {0:.4g} from beta={1:.4g}, z={2:.4g}".format(
                move, beta, z))
    return True

def scale_move_limit(filter_kw: Dict,
                     optimizer_kw: Dict,
                     move_min: float = 1e-3,
                     move_max: float = 0.2,
                     beta0: float = 1,
                     logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                     **kwargs: Any) -> None:
    """
    Adapt the MMA move limit depending on the current value of beta.

    Parameters
    ----------
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``; ``"eta"`` is
        read if present (default 0.5).
    optimizer_kw : dict
        MMA keyword dictionary. ``"move"`` is updated in-place.
    transition_eps : float
        projection value used to define the transition zone boundaries;
        the active interval is where ``transition_eps < H < 1 - transition_eps``
        (default 0.01).
    varsigma : float
        scaling divisor converting the zone width to the move limit
        (default 5.0).
    move_min : float
        lower clip bound on the move limit (default 0.01).
    move_max : float
        upper clip bound on the move limit (default 0.2).
    logger : BaseLogger
        logger object.
    
    Returns
    -------
    True : bool
        this function does not give any criteria about converging, so it is
        always ready to stop.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    optimizer_kw["move"] = np.maximum(move_max * beta0/filter_kw["beta"], move_min)
    logger.info("move limit adapted to {0:.4g} from beta={1:.4g}".format(
                optimizer_kw["move"], filter_kw["beta"]))
    return True

def scaling(change: float,
            loop: int,
            key: str,
            filter_kw: Dict,
            conv_tol: float,
            scale: float = 2.0,
            limit: float = 64.0,
            update: int = 50,
            state: Union[None, Dict] = None,
            logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
            **kwargs: Any) -> bool:
    """
    Continuation scheme for the Heaviside projection parameter beta.

    Increases ``filter_kw[key]`` by scaling by ``scale`` whenever the 
    change metric falls below ``conv_tol`` or the iteration counter since the 
    last beta update reaches ``key_update``.  Signals termination once
    ``limit`` is reached and the change is below ``conv_tol``.

    The iteration counter is maintained in-place in ``state[key+"_loop"]``,
    which is initialised automatically on the first call if ``state`` is an
    empty dict.  Pass the same dict object every call so state persists.

    Parameters
    ----------
    change : float
        current value of the convergence metric.
    loop : int
        current iteration index.
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``, updated in-place.
    conv_tol : float
        convergence tolerance. Continuation is triggered when
        ``change < conv_tol``.
    beta_scale : float
        multiplicative factor applied to ``filter_kw["beta"]`` at each
        continuation step (default 2.0).
    beta_limit : float
        upper bound on beta; terminates once reached and converged
        (default 64.0).
    beta_update : int
        maximum number of iterations between forced beta updates, even if
        convergence has not been reached (default 50).
    state : dict or None
        mutable dict for persisting the per-call iteration counter
        ``"beta_loop"``.  Pass the same dict every call; it is initialised
        automatically if empty or None.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    stop : bool
        True if continuation finished and design converged.
    """
    if "beta" not in filter_kw:
        raise ValueError(f'"{key}" is not in filter_kw: ', filter_kw.keys())
    if state is None:
        state = {}
    state.setdefault(key+"_loop", 0)
    state[key+"_loop"] += 1
    if change < conv_tol and \
       filter_kw[key] >= limit:
        stop = True
    elif (change < conv_tol or \
          state["beta_loop"] >= update) and \
          filter_kw["beta"] < limit:
        filter_kw[key] *= scale
        state[key+"_loop"] = 0
        logger.info("{0} increased.: {1: .1f}".format(key,filter_kw[key]))
        stop = False
    else:
        stop = False
    return stop

def beta_scaling(change: float,
                 loop: int,
                 filter_kw: Dict,
                 conv_tol: float,
                 beta_scale: float = 2.0,
                 beta_limit: float = 64.0,
                 beta_update: int = 50,
                 state: Union[None, Dict] = None,
                 logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                 **kwargs: Any) -> bool:
    """
    Continuation scheme for the Heaviside projection parameter beta.

    Increases ``filter_kw["beta"]`` by scaling by ``beta_scale`` whenever the 
    change metric falls below ``conv_tol`` or the iteration counter since the 
    last beta update reaches ``beta_update``.  Signals termination once
    ``beta_limit`` is reached and the change is below ``conv_tol``.

    The iteration counter is maintained in-place in ``state["beta_loop"]``,
    which is initialised automatically on the first call if ``state`` is an
    empty dict.  Pass the same dict object every call so state persists.

    Parameters
    ----------
    change : float
        current value of the convergence metric.
    loop : int
        current iteration index.
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``, updated in-place.
    conv_tol : float
        convergence tolerance. Continuation is triggered when
        ``change < conv_tol``.
    beta_scale : float
        multiplicative factor applied to ``filter_kw["beta"]`` at each
        continuation step (default 2.0).
    beta_limit : float
        upper bound on beta; terminates once reached and converged
        (default 64.0).
    beta_update : int
        maximum number of iterations between forced beta updates, even if
        convergence has not been reached (default 50).
    state : dict or None
        mutable dict for persisting the per-call iteration counter
        ``"beta_loop"``.  Pass the same dict every call; it is initialised
        automatically if empty or None.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    stop : bool
        True if continuation finished and design converged.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    if state is None:
        state = {}
    state.setdefault("beta_loop", 0)
    state["beta_loop"] += 1
    if change < conv_tol and \
       filter_kw["beta"] >= beta_limit:
        stop = True
    elif (change < conv_tol or \
          state["beta_loop"] >= beta_update) and \
          filter_kw["beta"] < beta_limit:
        filter_kw["beta"] *= beta_scale
        state["beta_loop"] = 0
        logger.info("beta increased.: {0: .1f}".format(filter_kw["beta"]))
        stop = False
    else:
        stop = False
    return stop

def beta_translation(change: float,
                     loop: int,
                     filter_kw: Dict,
                     conv_tol: float,
                     beta_translation: float = 1.0,
                     beta_limit: float = 64.0,
                     beta_update: int = 50,
                     state: Union[None, Dict] = None,
                     logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                     **kwargs: Any) -> bool:
    """
    Continuation scheme for the Heaviside projection parameter beta.

    Increases ``filter_kw["beta"]`` by adding ``beta_translation`` 
    whenever the change metric falls below ``conv_tol`` or the iteration 
    counter since the last beta update reaches ``beta_update``.  Signals 
    termination once ``beta_limit`` is reached and the change is below 
    ``conv_tol``.

    The iteration counter is maintained in-place in ``state["beta_loop"]``,
    which is initialised automatically on the first call if ``state`` is an
    empty dict.  Pass the same dict object every call so state persists.

    Parameters
    ----------
    change : float
        current value of the convergence metric.
    loop : int
        current iteration index.
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``, updated in-place.
    conv_tol : float
        convergence tolerance. Continuation is triggered when
        ``change < conv_tol``.
    beta_translation : float
        multiplicative factor applied to ``filter_kw["beta"]`` at each
        continuation step (default 2.0).
    beta_limit : float
        upper bound on beta; terminates once reached and converged
        (default 64.0).
    beta_update : int
        maximum number of iterations between forced beta updates, even if
        convergence has not been reached (default 50).
    state : dict or None
        mutable dict for persisting the per-call iteration counter
        ``"beta_loop"``.  Pass the same dict every call; it is initialised
        automatically if empty or None.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    stop : bool
        True if continuation finished and design converged.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    if state is None:
        state = {}
    state.setdefault("beta_loop", 0)
    state["beta_loop"] += 1
    if change < conv_tol and \
       filter_kw["beta"] >= beta_limit:
        stop = True
    elif (change < conv_tol or \
          state["beta_loop"] >= beta_update) and \
          filter_kw["beta"] < beta_limit:
        filter_kw["beta"] += beta_translation
        state["beta_loop"] = 0
        logger.info("beta increased.: {0: .1f}".format(filter_kw["beta"]))
        stop = False
    else:
        stop = False
    return stop

def dunning_beta_continuation(change: float,
                              loop: int,
                              filter_kw: Dict,
                              conv_tol: float,
                              gamma: float = 1e-4,
                              r: float = 0.2,
                              beta_limit: float = 512.0,
                              gray_tol: float = 0.01,
                              eps: float = 1e-16,
                              obj_hist: Union[None, List] = None,
                              xPhys_hist: Union[None, List] = None,
                              state: Union[None, Dict] = None,
                              logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                              **kwargs: Any) -> bool:
    """
    Adaptive beta continuation based on objective progress from:

    Dunning, P., & Wein, F. (2025). Automatic projection parameter increase for 
    three-field density-based topology optimization. Structural and 
    multidisciplinary optimization, 68(2), 33.

    This function derives the beta step from the recent change in the objective:

        d_beta = max(-0.5 * gamma * (f_k + f_{k-1}) / (f_k - f_{k-1}), 0)
        beta   = min(beta + min(d_beta, r * beta), beta_limit)

    Termination is signalled when beta has reached ``beta_limit`` and
    ``change < conv_tol``, or when the grayness of the projected design
    (measured via ``gray_indicator``) drops below ``gray_tol`` and
    ``change < conv_tol``.

    Parameters
    ----------
    change : float
        current value of the convergence metric.
    loop : int
        current iteration index.
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"``, updated in-place.
    conv_tol : float
        convergence tolerance used for the termination check.
    gamma : float
        objective-progress scaling factor (default 1e-4).
    r : float
        maximum relative beta step per iteration (default 0.2).
    beta_limit : float
        upper bound on beta (default 512.0).
    gray_tol : float
        grayness threshold for early termination (default 0.01).
    eps : float
        denominator guard against division by near-zero objective change
        (default 1e-16).
    obj_hist : list of float or None
        history of objective values; must contain at least two entries for
        the update to fire.
    xPhys_hist : list of np.ndarray or None
        history of physical density arrays; used to compute grayness for
        the termination check.
    state : dict or None
        unused; present for API consistency with ``beta_continuation``.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    stop : bool
        True if continuation finished and design converged.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    # grayness of the current projected design
    if xPhys_hist is not None and len(xPhys_hist) >= 1:
        gray = gray_indicator(xPhys_hist[-1])
    else:
        gray = None
    # termination checks
    if change < conv_tol:
        if (filter_kw["beta"] >= beta_limit) or\
            (gray is not None and gray <= gray_tol):
            stop = True
        else:
            stop=False
    else:
        stop = False
    # need at least two objective values to compute delta_beta
    if obj_hist is None or len(obj_hist) < 2:
        stop = False
    if not stop:
        obj_change = obj_hist[-1] - obj_hist[-2]
        # 
        if abs(obj_change) < eps:
            d_beta = r * filter_kw["beta"]
        # Eq. 6 in paper
        else:
            d_beta = max(-0.5 * gamma * (obj_hist[-1] + obj_hist[-2]) / obj_change, 0.)
        if not np.isfinite(d_beta):
            d_beta = 0.
        # combined update from Eq 7 with an added limit
        filter_kw["beta"] = np.minimum(filter_kw["beta"] + min(d_beta, r * filter_kw["beta"]),
                                       beta_limit)
        logger.info("beta increased: {0:.4g}, gray={1}, change={2:.4g}".format(
                    filter_kw["beta"], gray, change))
    return stop

def level_beta_continuation(change: float,
                            loop: int,
                            filter_kw: Dict,
                            conv_tol: float,
                            gamma: float = 1e-4,
                            r: float = 0.2,
                            beta_limit: float = 512.0,
                            gray_tol: float = 0.01,
                            eps: float = 1e-16,
                            obj_hist: Union[None, List] = None,
                            xPhys_hist: Union[None, List] = None,
                            state: Union[None, Dict] = None,
                            logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                            **kwargs: Any) -> bool:
    """
    Generalization of adaptive beta continuation for multi-level designs, based on
    objective progress from:

    Dunning, P., & Wein, F. (2025). Automatic projection parameter increase for
    three-field density-based topology optimization. Structural and
    multidisciplinary optimization, 68(2), 33.

    The beta increment is derived from the recent change in the objective (Eq. 6):

        d_beta = max(-0.5 * gamma * (f_k + f_{k-1}) / (f_k - f_{k-1}), 0)
        beta   = min(beta + min(d_beta, r * beta), beta_limit)

    If the absolute objective change is below ``eps``, the fallback
    ``d_beta = r * beta`` is used to guarantee progress.

    Unlike ``adaptive_beta_continuation``, discreteness is measured via
    ``level_indicator`` with levels ``[0] + filter_kw["weights"]``, reflecting
    the multi-level nature of the design. Termination is signalled when
    ``change < conv_tol`` and either beta has reached ``beta_limit`` or the
    level indicator drops below ``gray_tol``.

    Parameters
    ----------
    change : float
        current value of the convergence metric.
    loop : int
        current iteration index.
    filter_kw : dict
        filter keyword dictionary. Must contain ``"beta"`` and ``"weights"``,
        updated in-place.
    conv_tol : float
        convergence tolerance used for the termination check.
    gamma : float
        objective-progress scaling factor (default 1e-4).
    r : float
        maximum relative beta step per iteration, also used as fallback step
        when objective change is negligible (default 0.2).
    beta_limit : float
        upper bound on beta (default 512.0).
    gray_tol : float
        level-indicator threshold for early termination (default 0.01).
    eps : float
        guard against division by near-zero objective change (default 1e-16).
    obj_hist : list of float or None
        history of objective values; must contain at least two entries for
        the update to fire.
    xPhys_hist : list of np.ndarray or None
        history of physical density arrays; used to compute the level
        indicator for the termination check.
    state : dict or None
        unused; present for API consistency with other continuation functions.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    stop : bool
        True if continuation finished and design converged.
    """
    if "beta" not in filter_kw:
        raise ValueError('"beta" is not in filter_kw: ', filter_kw)
    # grayness of the current projected design
    if xPhys_hist is not None and len(xPhys_hist) >= 1:
        gray = level_indicator(xPhys_hist[-1], 
                               x_i=np.append(np.zeros(1), 
                                             filter_kw["weights"]))
    else:
        gray = None
    # termination checks
    if change < conv_tol:
        if (filter_kw["beta"] >= beta_limit) or\
            (gray is not None and gray <= gray_tol):
            stop = True
        else:
            stop=False
    else:
        stop = False
    # need at least two objective values to compute delta_beta
    if obj_hist is None or len(obj_hist) < 2:
        stop = False
    if not stop:
        obj_change = obj_hist[-1] - obj_hist[-2]
        # 
        if abs(obj_change) < eps:
            d_beta = r * filter_kw["beta"]
        # Eq. 6 in paper
        else:
            d_beta = max(-0.5 * gamma * (obj_hist[-1] + obj_hist[-2]) / obj_change, 0.)
        if not np.isfinite(d_beta):
            d_beta = 0.
        # combined update from Eq 7 with an added limit
        filter_kw["beta"] = np.minimum(filter_kw["beta"] + min(d_beta, r * filter_kw["beta"]),
                                       beta_limit)
        logger.info("beta increased: {0:.10g}, gray={1}, change={2:.4g}".format(
                    filter_kw["beta"], gray, change))
    return stop
