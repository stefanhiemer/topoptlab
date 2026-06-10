# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Dict,Tuple,Union
from functools import partial

import numpy as np
from scipy.optimize import root_scalar, minimize, Bounds,\
                           LinearConstraint, NonlinearConstraint

from topoptlab.log_utils import EmptyLogger,SimpleLogger

def find_eta(eta0: float, 
             xTilde: np.ndarray, 
             beta: float, 
             volfrac: float,
             root_args: Dict = {"fprime": True,
                                "fprime2": True,
                                "method": "halley",
                                "maxiter": 1000,
                                "bracket": [-1/2,1/2]},
            logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
             **kwargs: Any) -> float:
    """
    Find volume preserving eta for the element-wiser elaxed Haeviside 
    as has been done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta0 : float
        initial guess for threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.
    logger : BaseLogger
        logger object to log performance.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    # unfortunately scipy.optimize needs f to change sign between the
    # respective ends of the brackets, therefor the eta found by this function
    # is offset by -1/2 to the value later used
    result = root_scalar(f=_find_eta_root_func, 
                         x0=eta0-1/2, 
                         args=(xTilde,beta,volfrac),
                         x1=0.,
                         **root_args)
    #
    if result.converged:
        logger.perf("find_eta iterations=%d function_calls=%d", 
                    result.iterations, result.function_calls)
        return result.root+1/2
    else:
        raise ValueError("volume conserving eta could not be found: ",result)

def _find_eta_root_func(eta: float, 
                        xTilde: np.ndarray, 
                        beta: float, 
                        volfrac: float) -> Tuple[float,float]:
    """
    Function whose root is the volume preserving threshold.

    Parameters
    ----------
    eta : float
        current threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.

    Returns
    -------
    res : float
        value of current volume fraction - intended volume fraction.
    gradient : float
        gradient for Newton procedure

    """
    #
    eta = eta + 1/2
    #
    xProj = eta_projection(eta=eta,xTilde=xTilde,beta=beta)
    #
    return xProj.mean()-volfrac, \
           eta_projection_deta(eta = eta, 
                               xTilde=xTilde, 
                               xProj=xProj,
                               beta=beta).mean(), \
            eta_projection_deta2(eta = eta, 
                                 xTilde=xTilde, 
                                 xProj=xProj,
                                 beta=beta).mean()

def eta_projection(eta: float, 
                   xTilde: np.ndarray, 
                   beta: float) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    return (np.tanh(beta * eta) + np.tanh(beta * (xTilde - eta))) / \
           (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def eta_projection_dx(eta: float, 
                      xTilde: np.ndarray, 
                      beta: float) -> np.ndarray:
    """
    Perform first derivative of differentiable "relaxed" Haeviside projection as 
    done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity

    Returns
    -------
    xProj_dx : np.ndarray
        first derivative of projected densities.

    """
    return beta * (1 - np.tanh(beta * (xTilde - eta))**2) /\
                  (np.tanh(beta*eta)+np.tanh(beta*(1-eta))) 

def eta_projection_deta(eta: float, 
                        xTilde: np.ndarray, 
                        xProj: np.ndarray,
                        beta: float) -> np.ndarray:
    """
    Perform first derivative of differentiable "relaxed" Haeviside projection as 
    done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    xProj : np.ndarray
        eta-projected densities, i.e. the output of
        ``eta_projection(eta, xTilde, beta)``. Passed in to avoid
        recomputing the projection, since it appears in the derivative
        formula directly.
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity.

    Returns
    -------
    xProj_deta : np.ndarray
        derivative of projected densities with respect to eta.

    """
    return beta * ((1-xProj)*np.cosh(beta*eta)**(-2) -\
                   np.cosh(beta*(xTilde-eta))**(-2) + \
                   xProj*np.cosh(beta*(1-eta))**(-2)) /\
                  (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))

def eta_projection_deta2(eta: float, 
                         xTilde: np.ndarray, 
                         xProj: np.ndarray,
                         beta: float) -> np.ndarray:
    """
    Second derivative of the differentiable "relaxed" Haeviside projection
    with respect to eta, as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    eta : float
        threshold value.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    xProj : np.ndarray
        eta-projected densities, i.e. the output of
        ``eta_projection(eta, xTilde, beta)``. Passed in to avoid
        recomputing the projection, since it appears in the derivative
        formula directly.
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity.

    Returns
    -------
    xProj_deta2 : np.ndarray
        second derivative of projected densities with respect to eta.

    """
    xProj_deta = eta_projection_deta(eta=eta, 
                                     xTilde=xTilde, 
                                     xProj=xProj, 
                                     beta=beta)
    # term1 = (1-xProj)*np.cosh(beta*eta)**(-2)
    term1_deta = (-1)*(xProj_deta + 2*beta*(1-xProj)*np.tanh(beta*eta))*np.cosh(beta*eta)**(-2)
    # term2 = np.cosh(beta*(xTilde-eta))**(-2)
    term2_deta = 2*beta*np.cosh(beta*(xTilde-eta))**(-2) * np.tanh(beta*(xTilde-eta))
    # term3 = xProj*np.cosh(beta*(1-eta))**(-2)
    term3_deta = (xProj_deta + 2*beta*xProj*np.tanh(beta*(1-eta)))*np.cosh(beta*(1-eta))**(-2)
    # term4 = (np.tanh(beta*eta)+np.tanh(beta*(1-eta)))**(-1)
    term4_deta = (-beta)*(np.tanh(beta*eta)+np.tanh(beta*(1-eta)))**(-2) * \
                 (np.cosh(beta*eta)**(-2)-np.cosh(beta*(1-eta))**(-2))
    return beta * ((term1_deta -term2_deta+term3_deta)/(np.tanh(beta*eta)+np.tanh(beta*(1-eta)))+\
                    xProj_deta*term4_deta*(np.tanh(beta*eta)+np.tanh(beta*(1-eta))))

def find_multieta(etas0: Union[float,np.ndarray], 
                  xTilde: np.ndarray, 
                  beta: float, 
                  volfrac: float,
                  weights: np.ndarray,
                  mode: str = "mse",
                  etas_fixed: Union[None,np.ndarray] = None,
                  root_args: Dict = {"fprime": True,
                                     "method": "newton",
                                     "maxiter": 1000,
                                     "bracket": [-1/2,1/2]},
                  logger: Union[EmptyLogger,SimpleLogger] = EmptyLogger(),
                  **kwargs: Any) -> float:
    """
    Find volume preserving eta multiple eta projections

    Parameters
    ----------
    etas0 : np.ndarray
        initial guess for threshold values.
    xTilde : np.ndarray
        intermediate densities.
    beta : np.ndarray
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    volfrac : float
        volume fraction.
    weights : np.ndarray
        weights for combining the multiple threshold projections.
    root_args : dict
        arguments for root finding algorithm to find the volume conserving eta.

    Returns
    -------
    eta : float
        volume conserving eta.

    """
    #
    if mode == "fixed":
        etas0=np.hstack((etas_fixed, 0.5))
        func = None#_find_multieta_fixed
    elif mode == "equal":
        etas0=np.linspace(0, 1, weights.shape[0]+2)[1:-1]
        func = None#_find_multieta_equalspaced
    elif mode == "mse":
        etas0 = np.asarray(etas0, dtype=float)
        func = mean_squared_error
    else:
        raise ValueError(f"mode must be 'fixed','equal' or 'mse'. Current value {mode}")
    # scalar problems
    if mode in ["fixed","equal"]:
        raise NotImplementedError("Not yet implemented.")
        # root_scalar needs f to change sign between the respective ends of the
        # brackets, therefor the eta found by this function is offset by -1/2 to the value later used
        result = root_scalar(f=func,
                             x0=etas0-1/2,
                             args=(xTilde,
                                   beta,
                                   weights,
                                   volfrac),
                             x1=0.,
                             fprime=True,
                             method="newton",
                             maxiter=1000,
                             bracket=[-1/2,1/2])
        if result.converged:
            result = result.root + 1/2
        else:
            raise ValueError("volume conserving eta could not be found: ", result)
    elif mode in ["mse"]:
        #
        n = etas0.shape[0]
        constraints = []
        # bounds: each eta in (eps, 1-eps)
        eps_bnd = 1e-8
        bounds = Bounds(eps_bnd * np.ones(n), (1 - eps_bnd) * np.ones(n))
        # ordering constraint: eta[i+1] - eta[i] >= eps_bnd
        if n > 1:
            #
            A = np.zeros((n - 1, n))
            A[:,:-1] = -np.eye(n-1)
            A[:,1:] += np.eye(n-1)
            order_constraint = LinearConstraint(A, 
                                                lb=eps_bnd, 
                                                ub=np.inf)
            constraints.append(order_constraint)
        # volume constraint: mean(xPhys) == volfrac
        volume_constraint = NonlinearConstraint(fun=partial(volume_constraint_fun, 
                                                            xTilde=xTilde, 
                                                            beta=beta, 
                                                            weights=weights, 
                                                            volfrac=volfrac),#   volume_constraint_fun(etas, xTilde, beta, weights, volfrac),
                                                jac=partial(volume_constraint_jac,
                                                            xTilde=xTilde,
                                                            beta=beta,
                                                            weights=weights,
                                                            volfrac=volfrac),
                                                lb=-eps_bnd, 
                                                ub=eps_bnd)
        constraints.append(volume_constraint)
        result = minimize(fun=func,
                          x0=etas0,
                          args=(xTilde, beta, weights, volfrac),
                          method="trust-constr",
                          jac=mean_squared_error_jac,
                          hess=None,
                          bounds=bounds,
                          options={"maxiter": 10000, 
                                   "initial_tr_radius": 0.2},
                          constraints=constraints)
        #
        if result.success:
            logger.perf("find_multieta iterations=%d function_calls=%d jacobian_calls=%d", 
                        result.nit, result.nfev, result.njev)
            result = result.x 
        else:
            raise ValueError("volume conserving eta could not be found: ", result)
    #
    return result

def mean_squared_error(etas: np.ndarray, 
                       xTilde: np.ndarray, 
                       beta: float,
                       weights: Union[None,np.ndarray],
                       volfrac: float
                       ) -> Tuple[float,np.ndarray]:
    xProj = multieta_projection(etas=etas,
                                xTilde=xTilde,
                                beta=beta,
                                weights=weights)
    return ((xProj - xTilde)**2).mean()

def mean_squared_error_jac(etas: np.ndarray,
                           xTilde: np.ndarray,
                           beta: float,
                           weights: Union[None,np.ndarray],
                           volfrac: float
                           ) -> np.ndarray:
    xProj = multieta_projection(etas=etas,
                                xTilde=xTilde,
                                beta=beta,
                                weights=weights)
    dxProj_deta = multieta_projection_deta(etas=etas,
                                           xTilde=xTilde,
                                           beta=beta,
                                           weights=weights)
    return (2.0 * (xProj - xTilde)[..., None] * dxProj_deta).mean(axis=0)

def volume_constraint_fun(etas: np.ndarray, 
                          xTilde: np.ndarray, 
                          beta: float,
                          weights: Union[None,np.ndarray],
                          volfrac: float
                          ) -> Tuple[float,np.ndarray]:
    xProj = multieta_projection(etas=etas,
                                xTilde=xTilde,
                                beta=beta,
                                weights=weights)
    return xProj.mean() - volfrac

def volume_constraint_jac(etas: np.ndarray,
                          xTilde: np.ndarray,
                          beta: float,
                          weights: Union[None, np.ndarray],
                          volfrac: float) -> np.ndarray:
    return multieta_projection_deta(etas=etas,
                                    xTilde=xTilde,
                                    beta=beta,
                                    weights=weights).mean(axis=0)

def multieta_projection(etas: np.ndarray, 
                        xTilde: np.ndarray, 
                        beta: float, 
                        weights: Union[None,np.ndarray] = None,
                        **kwargs: Any
                        ) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    but with multiple thresholds.
    
    Parameters
    ----------
    etas : np.ndarray
        threshold values.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.

    Returns
    -------
    xProj : np.ndarray
        projected densities.

    """
    xProj = (np.tanh(beta*etas[None,...]) +\
             np.tanh(beta*(xTilde[...,None]-etas[None,...])))/\
            (np.tanh(beta*etas[None,...]) +\
             np.tanh(beta * (1 - etas[None,...])))
    if weights is None:
        weights = np.ones(etas.shape)/etas.shape[0]
    return np.sum( xProj * weights[None,...] ,axis=-1)

def multieta_projection_dx(etas: np.ndarray, 
                           xTilde: np.ndarray, 
                           beta: float, 
                           weights: Union[None,np.ndarray] = None
                           ) -> np.ndarray:
    """
    Perform a differentiable "relaxed" Haeviside projection as done in

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505
    
    but with multiple thresholds.
    
    Parameters
    ----------
    etas : np.ndarray
        threshold values.
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.  

    Returns
    -------
    xProj_dx : np.ndarray
        first derivative of projected densities.

    """
    xProj_dx = beta * (1 - np.tanh(beta * (xTilde[...,None] - etas[None,...]))**2) /\
                      (np.tanh(beta*etas[None,...])+np.tanh(beta*(1-etas[None,...])))
    
    if weights is None:
        weights = np.ones(etas.shape)/etas.shape[0]
    return np.sum( xProj_dx * weights[None,...] ,axis=-1)


def multieta_projection_deta(etas: np.ndarray,
                             xTilde: np.ndarray,
                             beta: float,
                             weights: Union[None,np.ndarray]
                             ) -> np.ndarray:
    """
    First derivative of the weighted multi-threshold Haeviside projection
    with respect to each threshold eta_n as a generalization to

    Xu S, Cai Y, Cheng G (2010) Volume preserving nonlinear density filter
    based on Heaviside functions. Struct Multidiscip Optim 41:495–505

    Parameters
    ----------
    etas : np.ndarray
        threshold values, shape (n_etas,).
    xTilde : np.ndarray
        intermediate densities (typically before a density filter is applied).
    beta : float
        sharpness factor. The higher the more we approach the Haeviside
        function which is recovered in the limit of beta to infinity
    weights : None or np.ndarray
        weights for combining the multiple threshold projections. If None,
        uniform weights are used.

    Returns
    -------
    xPhys_deta : np.ndarray
        d(xPhys)/d(eta_n) = w_n * d(xProj_n)/d(eta_n),
        shape (..., n_etas).

    """
    # individual projections: shape (..., N_etas)
    xProj_n = (np.tanh(beta * etas[None,...]) +\
               np.tanh(beta * (xTilde[...,None] - etas[None,...]))) /\
              (np.tanh(beta * etas[None,...]) +\
               np.tanh(beta * (1 - etas[None,...])))
    # 
    return beta * weights[None,...] * ( 
            (1 - xProj_n) * np.cosh(beta * etas[None,...])**(-2) -
             np.cosh(beta * (xTilde[...,None] - etas[None,...]))**(-2) +
             xProj_n * np.cosh(beta * (1 - etas[None,...]))**(-2)) \
             / (np.tanh(beta * etas) + np.tanh(beta * (1 - etas)))[None,...]

def guest_projection(x: np.ndarray,
                     beta: float,
                     **kwargs: Any) -> np.ndarray:
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

def guest_projection_dx(x: np.ndarray,
                        beta: float,
                        **kwargs: Any) -> np.ndarray:
    """
    Implements first derivative of the Haeviside projection by

    Guest, James K., Jean H. Prévost, and Ted Belytschko. "Achieving minimum
    length scale in topology optimization using nodal design variables and
    projection functions." International journal for numerical methods in
    engineering 61.2 (2004): 238-254.

    This projection is a smooth version of the Haeviside step function Theta(x),
    so in simple words, this projection sets every value that is larger than
    zero to one and everything smaller/equal to zero to zero. The filter
    equation is

        x_filtered = 1 - exp(-beta x) + x exp(-beta)

    so the first derivative is 

        dx_filtered = beta*exp(-beta*x) + exp(-beta)

    beta is the projection strength, that is typically ramped up during the TO
    process to large values. The larger beta, the closer this filter is to a
    Haeviside function.
    
    Parameters
    ----------
    x : np.ndarray
        (intermediate) design variables.
    beta : float
        projection strength.

    Returns
    -------
    dx_filtered : np.ndarray
        first derivative of filtered design variables.

    """
    return beta*np.exp(-beta*x) + np.exp(-beta)

def sigmund2007_projection(x: np.ndarray,
                           beta: float,
                           **kwargs: Any) -> np.ndarray:
    """
    Implements the Haeviside projection by

    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.

    This projection is a smooth version of the Haeviside step function Theta(1-x),
    so in simple words, this projection sets every value that is smaller than 
    one to zero and everything smaller/equal to one to one. The filter
    equation is

    x_filtered = np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)

    beta is the projection strength, that is typically ramped up during the TO
    process to large values. The larger beta, the closer this filter is to a
    Haeviside function.
    
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

def sigmund2007_projection_dx(x: np.ndarray,
                              beta: float,
                              **kwargs: Any) -> np.ndarray:
    """
    Implements first derivative of the Haeviside projection by

    Sigmund, Ole. "Morphology-based black and white filters for topology 
    optimization." Structural and Multidisciplinary Optimization 33.4 (2007): 
    401-424.

    This projection is a smooth version of the Haeviside step function Theta(1-x),
    so in simple words, this projection sets every value that is smaller than one 
    to zero and everything smaller/equal to one to one. The filter equation is

        x_filtered = np.exp(beta*(x-1)) - (1-x)*np.exp(-beta)

    so the first derivative is 

        dx = beta*exp(beta*(x-1)) + exp(-beta)

    beta is the projection strength, that is typically ramped up during the TO
    process to large values. The larger beta, the closer this filter is to a
    Haeviside function.
    
    Parameters
    ----------
    x : np.ndarray
        (intermediate) design variables.
    beta : float
        projection strength.

    Returns
    -------
    dx_filtered : np.ndarray
        first derivative of filtered design variables.

    """
    return np.exp(beta*(x-1)) * beta + np.exp(-beta)

if __name__ == "__main__":
    #
    eps = 1e-5
    #
    eta = 0.5
    beta = 10
    #
    xTilde = np.linspace(0,1.,11)[:-1][:,None]
    # finite difference
    xProj = eta_projection(eta=eta+eps,
                           xTilde=xTilde, 
                           beta=beta)
    xProj_deta = (eta_projection(eta=eta+eps, 
                                 xTilde=xTilde, 
                                 beta=beta) - \
                  eta_projection(eta=eta-eps, 
                                 xTilde=xTilde, 
                                 beta=beta))/(2*eps)
    print(xProj_deta)
    #
    xProj_deta = eta_projection_deta(xTilde=xTilde,
                                     eta=eta,
                                     xProj=xProj,
                                     beta=beta)
    print(xProj_deta)
    #
    etas = np.array([0.25, 0.5, 0.75])
    weights = np.array([1/3, 1/3, 1/3])
    #  
    xPhys = multieta_projection(etas=etas, xTilde=xTilde, beta=beta, weights=weights)
    print("xTilde shape :", xTilde.shape)
    print("xPhys  shape :", xPhys.shape)
    #
    xPhys_dx = multieta_projection_dx(etas=etas, xTilde=xTilde, beta=beta, weights=weights)
    xPhys_dx_fd = (multieta_projection(etas=etas, xTilde=xTilde+eps, beta=beta, weights=weights) -
                   multieta_projection(etas=etas, xTilde=xTilde-eps, beta=beta, weights=weights)) / (2*eps)
    print("xPhys_dx  shape (analytic) :", xPhys_dx.shape)
    print("xPhys_dx  shape (fd)       :", xPhys_dx_fd.shape)
    print("max abs err d/dx :", np.max(np.abs(xPhys_dx - xPhys_dx_fd)))
    #
    xPhys_deta = multieta_projection_deta(etas=etas, xTilde=xTilde, beta=beta, weights=weights)
    xPhys_deta_fd = np.zeros((*xTilde.shape, etas.shape[0]))
    for n in range(etas.shape[0]):
        etas_p = etas.copy(); etas_p[n] += eps
        etas_m = etas.copy(); etas_m[n] -= eps
        xPhys_deta_fd[..., n] = (multieta_projection(etas=etas_p, xTilde=xTilde,
                                                      beta=beta, weights=weights) -
                                  multieta_projection(etas=etas_m, xTilde=xTilde,
                                                      beta=beta, weights=weights)) / (2*eps)
    print("xPhys_deta shape (analytic) :", xPhys_deta.shape)
    print("xPhys_deta shape (fd)       :", xPhys_deta_fd.shape)
    print("max abs err d/detas :", np.max(np.abs(xPhys_deta - xPhys_deta_fd)))