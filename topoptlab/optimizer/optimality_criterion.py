# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np
from scipy.sparse import spdiags, sparray

projections = [2,3,4,5]
filters = [0,1]

def oc_model(x: np.ndarray,
             x0: np.ndarray, 
             f0: float | np.ndarray, 
             dfdx : np.ndarray) -> float | np.ndarray:
    """
    Evaluate the optimality-criteria (OC) surrogate of ``f`` at ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the OC convex separable approximation: components with 
    ``dfdx<0`` sensitivities are linearized in ``x**(-1)`` while all others are 
    linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    f0 : float or np.ndarray
        Value of the original function at ``x0``. Use a scalar for a single
        function or an array of shape ``(m,)`` for multiple functions.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.

    Returns
    -------
    f_surrogate : float or np.ndarray
        Surrogate value at ``x``. Returns a scalar for 1D inputs and an array
        of shape ``(m,)`` for 2D inputs.

    """
    return f0 + np.where(dfdx<0,
                         -dfdx*(1/x - 1/x0)*x0**2, 
                         dfdx*(x-x0)).sum(axis=0)

def oc_model_dx(x: np.ndarray,
                x0: np.ndarray, 
                dfdx : np.ndarray) -> np.ndarray:
    """
    Evaluate the first derivative / jacobian of the optimality-criteria (OC) 
    surrogate of ``f`` at ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the OC convex separable approximation: components with 
    ``dfdx<0`` sensitivities are linearized in ``x**(-1)`` while all others are 
    linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.

    Returns
    -------
    f_surrogate_dx : np.ndarray
        First derivative of the surrogate at ``x`` of shape ``(n)`` or 
        ``(n,m)``.

    """
    return np.where(dfdx<0,
                    dfdx*(x0/x)**2, 
                    dfdx)

def oc_model_dx2(x: np.ndarray,
                x0: np.ndarray, 
                dfdx : np.ndarray, 
                return_diagonal : bool = True) -> np.ndarray | list[sparray]:
    """
    Evaluate the second derivative / hessian of the optimality-criteria (OC) 
    surrogate of ``f`` at ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the OC convex separable approximation: components with 
    ``dfdx<0`` sensitivities are linearized in ``x**(-1)`` while all others are 
    linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.
    return_diagonal : bool
        if True, only the diagonals are returned.

    Returns
    -------
    f_surrogate_dx2 : list or np.ndarray
        list of sparse hessians of the surrogate at ``x`` of shape ``(n,n)``
        or just the diagonals of shape ``(n)`` or ``(n,m)``.

    """
    diag = np.where(dfdx<0,
                    -2.*dfdx*(x0/x)**2 / x, 
                     0.)
    if return_diagonal:
        return diag
    elif x.ndim == 1:
        return [spdiags(data=diag, 
                        k=0, m=x.shape[0], n=x.shape[0], format="dia")]
    elif x.ndim == 2:
        return [spdiags(data=diag[:,i], k=0, m=x.shape[0], n=x.shape[0], format="dia") \
                for i in range(x.shape[1])]
    else:
        raise ValueError("x must have shape (n,) or (n, m)")
           
def ocgeneralized_model(x: np.ndarray,
                        x0: np.ndarray, 
                        f0: float | np.ndarray, 
                        dfdx : np.ndarray, 
                        damp: float) -> float | np.ndarray:
    """
    Evaluate the generalized  optimality-criteria (OC) surrogate of ``f`` at 
    ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the OC generalized convex separable approximation: components 
    with ``dfdx<0`` sensitivities are linearized in ``x**(-damp)`` while all 
    others are linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    f0 : float or np.ndarray
        Value of the original function at ``x0``. Use a scalar for a single
        function or an array of shape ``(m,)`` for multiple functions.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.
    damp : float 
        exponent commonly referred as damping exponent.

    Returns
    -------
    f_surrogate: float or np.ndarray
        Surrogate value at ``x``. Returns a scalar for 1D inputs and an array
        of shape ``(m,)`` for 2D inputs.

    """
    return f0 + np.where(dfdx<0,
                         -dfdx*( x**(-damp) - x0**(-damp) )*x0**(1+damp) / damp, 
                         dfdx*(x-x0)).sum(axis=0)

def ocgeneralized_model_dx(x: np.ndarray,
                           x0: np.ndarray, 
                           dfdx : np.ndarray, 
                           damp : float) -> np.ndarray:
    """
    Evaluate the first derivative / jacobian of the generalized 
    optimality-criteria (OC) surrogate of ``f`` at ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the generalized OC convex separable approximation: components 
    with ``dfdx<0`` sensitivities are linearized in ``x**(-damp)`` while all 
    others are linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.
    damp : float 
        exponent commonly referred as damping exponent.

    Returns
    -------
    f_surrogate_dx : np.ndarray
        First derivative of the surrogate at ``x`` of shape ``(n)`` or 
        ``(n,m)``.

    """
    return np.where(dfdx<0,
                    dfdx*(x0/x)**(damp+1.), 
                    dfdx)

def ocgeneralized_model_dx2(x : np.ndarray,
                            x0 : np.ndarray, 
                            dfdx : np.ndarray,
                            damp : float,
                            return_diagonal : bool = True) -> np.ndarray | list[sparray]:
    """
    Evaluate the second derivative / hessian of the optimality-criteria (OC) 
    surrogate of ``f`` at ``x``.

    The surrogate is constructed by expanding ``f`` at the current iterate
    ``x0`` using the OC convex separable approximation: components with 
    ``dfdx<0`` sensitivities are linearized in ``x**(-damp)`` while all others are 
    linearized in ``x``.

    Parameters
    ----------
    x : np.ndarray
        Evaluation point for the surrogate. Shape ``(n,)`` for a single
        function or ``(n, m)`` for ``m`` functions evaluated in parallel.
    x0 : np.ndarray
        Expansion point, typically the current design iterate. Must have the
        same shape as ``x``.
    dfdx : np.ndarray
        Sensitivity of ``f`` with respect to ``x``, evaluated at ``x0``.
        Must have the same shape as ``x``.
    damp : float 
        exponent commonly referred as damping exponent.
    return_diagonal : bool
        if True, only the diagonals are returned.

    Returns
    -------
    f_surrogate_dx2 : list or np.ndarray
        list of sparse hessians of the surrogate at ``x`` of shape ``(n,n)``
        or just the diagonals of shape ``(n)`` or ``(n,m)``.

    """
    diag = np.where(dfdx<0,
                    -(damp+1)*dfdx*(x0/x)**(damp+1.) / x, 
                     0.)
    if return_diagonal:
        return diag
    elif x.ndim == 1:
        return [spdiags(data=diag, 
                        k=0, m=x.shape[0], n=x.shape[0], format="dia")]
    elif x.ndim == 2:
        return [spdiags(data=diag[:,i], k=0, m=x.shape[0], n=x.shape[0], format="dia") \
                for i in range(x.shape[1])]
    else:
        raise ValueError("x must have shape (n,) or (n, m)")

def oc_top88(x: np.ndarray, volfrac: float, 
             dc: np.ndarray, dv: np.ndarray, 
             g: float,
             el_flags: None | np.ndarray,
             move: float = 0.2,
             l1: float = 0.,
             l2: float = 1e9) -> tuple[np.ndarray,float]:
    """
    Optimality criteria method (section 2.2 in top88 paper) for maximum/minimum
    stiffness/compliance. Heuristic updating scheme for the element densities
    to find the Lagrangian multiplier. Overtaken and adapted from the

    165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN
    
    Only sufficient for pure sensitivity/density filter optionally with 
    Helmholtz PDE. Haeviside projections or any filters that introduce a 
    stronger nonlinearity cannot be dealt with.
    
    Parameters
    ----------
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    el_flags : np.ndarray or None
        array of flags/integers that switch behaviour of specific elements. 
        Currently 1 marks the element as passive (zero at all times), while 2
        marks it as active (1 at all time).
    move: float
        maximum change allowed in the density of a single element.
    l1: float
        starting guess for the lower part of the bisection algorithm.
    l2: float
        starting guess for the upper part of the bisection algorithm.
        
    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, 
                             np.maximum(x-move, 
                                        np.minimum(1.0, 
                                                   np.minimum(x+move, x*np.sqrt(-dc/dv/lmid)))))
        
        # passive element update
        if el_flags is not None:
            xnew[el_flags==1] = 0
            xnew[el_flags==2] = 1
        gt=g+np.sum(dv*(xnew-x))
        #gt = xnew.mean() > volfrac
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        
    return (xnew, gt)

def oc_haevi(x: np.ndarray,
             volfrac: float,
             dc: np.ndarray,
             dv: np.ndarray,
             g: float,
             pass_el: None | np.ndarray,
             H: Any,
             Hs: np.ndarray,
             beta: None | float,
             eta: float,
             ft: None | int,
             debug: bool | int = False,
             ) -> tuple[np.ndarray, float] | tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Optimality criteria method (section 2.2 in top88 paper) for maximum/minimum 
    stiffness/compliance. Heuristic updating scheme for the element densities 
    to find the Lagrangian multiplier. Overtaken and adapted from the 
    
    165 LINE TOPOLOGY OPTIMIZATION CODE BY NIELS AAGE AND VILLADS EGEDE JOHANSEN
    
    Only sufficient for pure sensitivity/density filter optionally with 
    Helmholtz PDE. 
    
    Parameters
    ----------
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    pass_el : None or np.array 
        array who contains indices used for un/masking passive elements. 0 
        means an active element that is part of the optimization, 1 and 2 
        indicate empty and full elements which are not part of the 
        optimization.

    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    l1 = 0
    l2 = 1e9
    if ft is None or ft in [0,1]:
        move = 0.2
        tol = 1e-3
    else:
        move = 0.2
        tol = 1e-3
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    xTilde = xnew.copy()
    xPhys = xnew.copy()
    if debug:
        i = 0
    while (l2-l1)/(l1+l2) > tol and np.abs(l2-l1) > 1e-10:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, 
                             np.maximum(x-move, 
                                        np.minimum(1.0, 
                                                   np.minimum(x+move, 
                                                              x*np.sqrt(-dc/dv/lmid)))))
        #
        if ft in projections:
            xTilde = np.asarray(H*xnew[np.newaxis].T/Hs)[:, 0]
        if ft in [2]:
            xPhys = 1 - np.exp(-beta*xTilde) + xTilde*np.exp(-beta)
        elif ft in [3]:
            xPhys = np.exp(-beta*(1-xTilde)) - (1-xTilde)*np.exp(-beta)
        elif ft in [4]:
            xPhys = (np.tanh(beta*eta)+np.tanh(beta * (xTilde - eta)))/\
                    (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))
        else:
            xPhys = xnew
        # passive element update
        if pass_el is not None:
            xPhys[pass_el==1] = 0
            xPhys[pass_el==2] = 1
        #
        if ft not in projections:
            gt = g+np.sum(dv*(xnew-x))
        else:
            gt = xPhys.mean() > volfrac
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        if debug == 2:
            i = i+1
            print(f"oc it.: {i} , l1: {l1:.10f} l2: {l2:.10f}, gt: {gt:.10f}",
                f"x: {np.median(x):.10f} xTilde: {np.median(xTilde):.10f} xPhys: {np.median(xPhys):.10f}",
                f"dc: {np.max(dc):.10f} dv: {np.min(dv):.10f}")
            if np.isnan(gt):
                print()
                import sys 
                sys.exit()
    if beta is None:
        return (xPhys, gt)
    else:
        return (xnew, xTilde, xPhys, gt)
    
def oc_mechanism(x: np.ndarray, volfrac: float, 
                 dc: np.ndarray, dv: np.ndarray, 
                 g: float,
                 el_flags: None | np.ndarray,
                 move: int = 0.1, damp: float = 0.3,
                 l1: float = 0.,l2: float = 1e9) -> tuple[np.ndarray,float]:
    """
    Optimality criteria method for compliant mechnanism according to the 
    standard textbook by Bendsoe and Sigmund. In general: can handle objective 
    functions whose gradients change sign, but no guarantee of convergence or 
    anything else is given.
    
    Parameters
    ----------
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    el_flags : None or np.array 
        array who contains indices used for un/masking passive elements. 0 
        means an active element that is part of the optimization, 1 and 2 
        indicate empty and full elements which are not part of the 
        optimization.

    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew[:] = np.maximum(0.0, np.maximum(
            x-move, np.minimum(1.0, np.minimum(x+move, x*np.maximum(1e-10,
                                                                    -dc/dv/lmid)**damp))))
        
        # passive element update
        if el_flags is not None:
            xnew[el_flags==1] = 0
            xnew[el_flags==2] = 1
        gt = xnew.sum() - volfrac * x.shape[0] #g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        
    return (xnew, gt)

def oc_generalized(x: np.ndarray, volfrac: float, 
                   dc: np.ndarray, dv: np.ndarray, 
                   g: float,
                   el_flags: None | np.ndarray,
                   move: int = 0.1, damp: float = 0.3,
                   l1: float = 0.,l2: float = 1e9) -> tuple[np.ndarray,float]:
    """
    This is a function where I try around various generalizations. At the 
    moment identical to oc_mechanism.
    
    Parameters
    ----------
    x : np.array, shape (nel)
        element densities for topology optimization of the current iteration.
    volfrac : float
        volume fraction.
    dc : np.array, shape (nel)
        gradient of objective function/complicance with respect to element 
        densities.
    dv : np.array, shape (nel)
        gradient of volume constraint with respect to element densities..
    g : float
        parameter for the heuristic updating scheme.
    el_flags : None or np.array 
        array who contains indices used for un/masking passive elements. 0 
        means an active element that is part of the optimization, 1 and 2 
        indicate empty and full elements which are not part of the 
        optimization.

    Returns
    -------
    xnew : np.array, shape (nel)
        updatet element densities for topology optimization.
    gt : float
        updated parameter for the heuristic updating scheme..

    """
    # reshape to perform vector operations
    xnew = np.zeros(x.shape)
    while (l2-l1)/(l1+l2) > 1e-4 and l2 > 1e-40:
        lmid = 0.5*(l2+l1)
        xnew = np.maximum(0.,
                          np.maximum(x-move, 
                                     np.minimum(1., 
                                                np.minimum(x+move, 
                                                           x*np.maximum(1e-10,
                                                                        (-dc)/dv/lmid)**damp))))
        # passive element update
        if el_flags is not None:
            xnew[el_flags==1] = 0.
            xnew[el_flags==2] = 1.
        gt = xnew.sum() - volfrac * x.shape[0] #g+np.sum((dv*(xnew-x)))
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
        
    return (xnew, gt)
