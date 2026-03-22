from numpy import array,sqrt,zeros,ones
from numpy.random import rand,seed
from numpy import abs as npabs
from numpy.testing import assert_almost_equal
from scipy.optimize import rosen, rosen_der

import pytest

from topoptlab.optimizer.augmented_lagrangian import alm_first_order

@pytest.mark.parametrize('nvars,start_constrained',
                         [(3,True),])

def test_alm(nvars: int,
             start_constrained: bool,
             rho: float = 1e0,
             rho_scale: float = 1.05,
             maxiter: int = 2000) -> None:
    """
    Demonstrate the naive use of the first-order augmented-Lagrangian method 
    on the Rosenbrock function on the interval [-1.5, 1.5] with the equality
    constraint

        sum(x_i^2) / nvars - 1 = 0.

    The constrained problem is

        minimize    rosen(x)
        subject to  (x**2).sum()/nvars - 1 = 0
                    -1.5 <= x_i <= 1.5.
    
    This demonstration uses one equality constraint and no inequality
    constraints. Empty arrays are passed for the inequality terms in order to
    match the interface of ``alm_first_order``.

    Parameters
    ----------
    nvars : int, optional
        Number of design variables.
    maxiter : int, optional
        Maximum number of outer iterations.
    start_constrained : bool, optional
        If True, project the initial guess onto the equality constraint.
    rho : float, optional
         penalty parameter.
    """
    #
    seed(1)
    #
    x = rand(nvars, 1)
    #
    if start_constrained:
        x *= sqrt(nvars / (x**2).sum())
    xold = x.copy()
    xnew = zeros(x.shape)
    #
    fgradold = None
    ceqold = 1e5
    # lagr. multiplier
    lam = zeros((1, 1))
    mu = zeros((0, 1))
    #
    xmin = -1.5 * ones((nvars, 1))
    xmax = 1.5 * ones((nvars, 1))
    #
    move = 1e-2
    print(x)
    #
    for i in range(maxiter):
        #
        obj = rosen(x)[0]
        fgrad = rosen_der(x)
        # equality constraint: (x^2).sum()/nvars - 1 = 0
        ceq = array([(x**2).sum() / nvars - 1.])
        dceq = (2. / nvars) * x
        # no inequality constraints in this example
        cineq = zeros((0, 1))
        dcineq = zeros((nvars,0))
        #
        xnew, lam[:], _ = alm_first_order(x=x[:,0],
                                          fgrad=fgrad[:,0],
                                          xold=xold[:,0],
                                          fgradold=fgradold,
                                          ceq=ceq,
                                          dceq=dceq,
                                          cineq=cineq,
                                          dcineq=dcineq,
                                          lam=lam,
                                          mu=mu,
                                          xmin=xmin[:,0],
                                          xmax=xmax[:,0],
                                          rho=rho,
                                          move=move)
        # 
        if abs(ceq) < abs(ceqold):
            rho = rho * rho_scale
        #
        xold[:] = x
        fgradold = fgrad[:,0]\
                   + dceq.dot(lam + rho * ceq)[:,0]\
                   + dcineq.dot(mu + rho * cineq)[:,0]
        ceqold = ceq 
        #
        x[:,0] = xnew
        #
        change = npabs(x - xold).max()
        #
        if change <= 1e-7:
            break
    #
    print(x)
    assert_almost_equal(obj,0)

    return 