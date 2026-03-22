# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Tuple, Union

import numpy as np

from topoptlab.optimizer.gradient_descent import gradient_descent
from topoptlab.optimizer.stepsize import barzilai_borwein_short


def alm_first_order(x: np.ndarray,
                    fgrad: np.ndarray,
                    xold: np.ndarray,
                    fgradold: np.ndarray,
                    ceq: np.ndarray,
                    dceq: np.ndarray,
                    cineq: np.ndarray,
                    dcineq: np.ndarray,
                    lam: np.ndarray,
                    mu: np.ndarray,
                    xmin: Union[float, np.ndarray],
                    xmax: Union[float, np.ndarray],
                    rho: float,
                    stepsize_func: Callable = barzilai_borwein_short,
                    move: float = 0.1,
                    **kwargs: Any, 
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one first-order augmented-Lagrangian update for equality and
    inequality constraints.

    The method considers the constrained optimization problem

        minimize    f(x)
        subject to  ceq(x) = 0
                    cineq(x) <= 0

    with the Lagrangian
        
        L = f(x) + lam.T@ceq + mu.T@cineq
        
    which is augmented by a penalty term to
        
        L_a = f(x) 
              + lam.T@ceq + rho*lam.T@(ceq**2) 
                  + mu.T@cineq + rho*mu.T@(cineq**2)
    
    It uses a first-order primal step based on the gradient of the augmented
    Lagrangian, followed by standard multiplier updates.
    
    The primal gradient is assembled as

        grad_x L_a = fgrad
                     + dceq.T @ (lam + rho * ceq)
                     + dcineq.T @ (mu + rho * max(cineq, 0)).

    The inequality multipliers are updated by 

        mu_new = max(0, mu + rho * cineq)
        
    whereas the equality multipliers by

        lam_new = lam + rho * cineq.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        Design variables of the current iteration.
    fgrad : np.ndarray, shape (n,)
        Gradient of the objective function at the current iteration.
    xold : np.ndarray, shape (n,)
        Design variables of the previous iteration.
    fgradold : np.ndarray, shape (n,)
        previous gradient used in the primal descent step. When using
        Barzilai-Borwein step-size rules, this should be the gradient of the
        augmented Lagrangian from the previous iteration, not only the bare
        objective gradient.
    ceq : np.ndarray, shape (meq,)
        Equality-constraint values at the current iteration.
    dceq : np.ndarray, shape (n,meq)
        jacobian of the equality constraints with respect to the design
        variables. Row ``i`` contains the gradient of constraint ``ceq[i]``.
    cineq : np.ndarray, shape (mineq,)
        inequality-constraint values at the current iteration. Feasible values
        satisfy ``cineq <= 0`` componentwise.
    dcineq : np.ndarray, shape (n,mineq)
        jacobian of the inequality constraints with respect to the design
        variables. Row ``i`` contains the gradient of constraint ``cineq[i]``.
    lameq : np.ndarray, shape (meq,)
        current Lagrange multipliers for the equality constraints.
    lamineq : np.ndarray, shape (mineq,)
        current Lagrange multipliers for the inequality constraints.
    xmin : float or np.ndarray, shape (n,)
        lower bounds for the design variables.
    xmax : float or np.ndarray, shape (n,)
        upper bounds for the design variables.
    rho : float
        augmented-Lagrangian penalty parameter.
    stepsize_func : Callable, optional
        function used to determine the primal step size.
    move : float, optional
        maximum change allowed in each design variable.
    **kwargs : Any
        additional keyword arguments passed to ``gradient_descent`` and, in
        turn, to the step-size function.

    Returns
    -------
    xnew : np.ndarray, shape (n,)
        updated design variables after one primal descent step.
    lameqnew : np.ndarray, shape (meq,)
        updated Lagrange multipliers for the equality constraints.
    lamineqnew : np.ndarray, shape (mineq,)
        updated Lagrange multipliers for the inequality constraints.

    """
    # update design variables
    xnew = gradient_descent(x=x,
                            dobj=fgrad\
                                 + dceq.dot(lam + rho * ceq)[:,0]\
                                 + dcineq.dot(mu + rho * cineq)[:,0],
                            xold=xold,
                            dobjold=fgradold,
                            xmin=xmin,
                            xmax=xmax,
                            stepsize_func=stepsize_func,
                            move=move)
    #
    return xnew, lam + rho * ceq, np.maximum(0.0, mu + rho * cineq)