# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.augmented_lagrangian import alm_first_order, alm_lagrangian
from topoptlab.optimizer.stepsize import constant,barzilai_borwein_short,barzilai_borwein_long
from topoptlab.optimizer.mma_utils import mma_defaultkws,gcmma_defaultkws
from mmapy import mmasub,gcmmasub, kktcheck, asymp, concheck, raaupdate

def demonstrate_alm_line(nvars=3,
                         verbose=False,
                         maxiter=2000,
                         maxinnerit=5,
                         start_constrained=True,
                         rho=1e0, 
                         rho_scale=2, 
                         eps_eq=1e-8):
    """
    Demonstrate the first-order augmented-Lagrangian method together with inner
    iterations for line search the Rosenbrock function on the interval 
    [-1.5, 1.5] with the equality constraint

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
    verbose : bool, optional
        Whether to print iteration information.
    maxiter : int, optional
        Maximum number of outer iterations.
    start_constrained : bool, optional
        If True, project the initial guess onto the equality constraint.
    rho : float, optional
         penalty parameter.

    Returns
    -------
    x : np.ndarray, shape (nvars, 1)
        Final design variables.
    fgrad : np.ndarray, shape (nvars, 1)
        Final objective gradient.
    lam : np.ndarray, shape (1, 1)
        Final equality multiplier.
    niter : int
        Number of iterations performed.
    """
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars, 1)
    if start_constrained:
        x *= np.sqrt(nvars / (x**2).sum())
    xold = x.copy()
    xnew = np.zeros(x.shape)
    #
    fgradold = None
    # lagr. multiplier
    lam = np.ones((1, 1))
    mu = np.ones((0, 1))
    #
    xmin = -1.5 * np.ones((nvars, 1))
    xmax = 1.5 * np.ones((nvars, 1))
    #
    move = 1e-2
    # evaluate everything for the start
    obj0 = rosen(x)[0]
    obj = 1
    fgrad = rosen_der(x)/obj0
    # equality constraint: (x^2).sum()/nvars - 1 = 0
    ceq = np.array([(x**2).sum() / nvars - 1.])
    ceqold = np.ones(ceq.shape) * np.inf
    dceq = (2. / nvars) * x
    # no inequality constraints in this example
    cineq = np.zeros((0, 1))
    dcineq = np.zeros((nvars,0))
    #
    lagr = alm_lagrangian(obj=obj, 
                          ceq=ceq,
                          cineq=cineq,
                          lam=lam,
                          mu=mu, 
                          rho=rho)
    if verbose:
        print("it.: {0}, obj.: {1:.10f}, lagr.: {2:.10f}, ceq.: {3:.10f}, "
              "rho: {4:.4e}, ch.: {5:.10f}".format(
                0,
                obj,
                lagr,
                ceq[0],
                rho,
                0.))
    for i in range(maxiter):
        #
        step_size = barzilai_borwein_long(x=x[:,0], 
                                           fgrad=fgrad[:,0]\
                                                 + dceq.dot(lam + rho * ceq)[:,0]\
                                                 + dcineq.dot(mu + rho * cineq)[:,0], 
                                           xold=xold[:,0], 
                                           fgradold=fgradold)
        # inner iterations for line search
        for j in range(maxinnerit):
            #
            if np.isclose(step_size, 0):
                raise RuntimeError("line search failed.")
            # trial update design variables
            xnew[:,0], lam_new, mu_new = alm_first_order(x=x[:,0],
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
                                               stepsize_func=constant,
                                               stepsize_kw={"step_size": step_size},
                                               rho=rho,
                                               move=move)
            # recalculate 
            obj_new = rosen(xnew)[0]/obj0
            # equality constraint: (x^2).sum()/nvars - 1 = 0
            ceq_new = np.array([(xnew**2).sum() / nvars - 1.])
            # no inequality constraints in this example
            cineq_new = np.zeros((0, 1))
            #
            lagr_new = alm_lagrangian(obj=obj_new, 
                                  ceq=ceq_new,
                                  cineq=cineq_new,
                                  lam=lam,
                                  mu=mu,
                                  rho=rho)
            # 
            change = np.abs(xnew-x).max()
            #
            if verbose and False:
                print("inner it.: {0}, obj.: {1:.10f}, lagr.: {2:.10f}, ceq.: {3:.10f}, "
                      "rho: {4:.4e}, ch.: {5:.10f}".format(
                        j+1,
                        obj_new,
                        lagr_new,
                        ceq_new[0],
                        rho,
                        change))
            # if constraint fulfilled or improved and obj decreased, pass
            if lagr_new < lagr:
                break
            # scale penalty 
            #ho = rho * rho_scale
            # reduce step size
            step_size=step_size * 3/4
        # store variables for memory/comparison
        xold[:] = x
        fgradold = fgrad[:,0]\
                   + dceq.dot(lam + rho * ceq)[:,0]\
                   + dcineq.dot(mu + rho * cineq)[:,0]
        lam = lam_new.copy()
        mu = mu_new.copy()
        # update 
        x[:] = xnew.copy()
        obj = obj_new.copy()
        ceqold = ceq.copy()
        ceq = ceq_new.copy()
        cineq = cineq_new.copy()
        lagr = lagr_new
        # scale penalty if equality constraint violated and not improving
        if (np.abs(ceq) > eps_eq) and (np.abs(ceq) > np.abs(ceqold)):
            rho = rho * rho_scale
        #
        change = np.abs(x - xold).max()
        #
        if verbose:
            print("it.: {0}, obj.: {1:.10f}, lagr.: {2:.10f}, ceq.: {3:.10f}, "
                  "rho: {4:.4e}, ch.: {5:.10f}".format(
                    i + 1,
                    obj,
                    lagr,
                    ceq[0],
                    rho,
                    change))
        #
        if (change <= 1e-7):
            break
        # evaluate gradients
        fgrad = rosen_der(x)/obj0
        # equality constraint: (x^2).sum()/nvars - 1 = 0
        dceq = (2. / nvars) * x
        # no inequality constraints in this example
        dcineq = np.zeros((nvars,0))
        
    print("Augmented Lagrangian Method")
    print("final objective: ", obj)
    print("final x: ", x[:, 0])
    print("final gradient: ", fgrad[:, 0])
    print("constraint vs ideal: ", (x**2).sum(), nvars)
    print("final lambda: ", float(lam[0, 0]))
    print("after {0} iterations".format(i + 1))

    return x, fgrad, lam, i + 1

def demonstrate_alm(nvars=3,
                    verbose=False,
                    maxiter=2000,
                    start_constrained=False,
                    rho=1e0, 
                    rho_scale=1.05, 
                    eps_eq=1e-8):
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
    verbose : bool, optional
        Whether to print iteration information.
    maxiter : int, optional
        Maximum number of outer iterations.
    start_constrained : bool, optional
        If True, project the initial guess onto the equality constraint.
    rho : float, optional
         penalty parameter.

    Returns
    -------
    x : np.ndarray, shape (nvars, 1)
        Final design variables.
    fgrad : np.ndarray, shape (nvars, 1)
        Final objective gradient.
    lam : np.ndarray, shape (1, 1)
        Final equality multiplier.
    niter : int
        Number of iterations performed.
    """
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars, 1)
    if start_constrained:
        x *= np.sqrt(nvars / (x**2).sum())
    xold = x.copy()
    xnew = np.zeros(x.shape)
    #
    fgradold = None
    ceqold = 1e5
    # lagr. multiplier
    lam = np.zeros((1, 1))
    mu = np.zeros((0, 1))
    #
    xmin = -1.5 * np.ones((nvars, 1))
    xmax = 1.5 * np.ones((nvars, 1))
    #
    move = 1e-2
    #
    for i in range(maxiter):
        #
        obj = rosen(x)[0]
        fgrad = rosen_der(x)
        # equality constraint: (x^2).sum()/nvars - 1 = 0
        ceq = np.array([(x**2).sum() / nvars - 1.])
        dceq = (2. / nvars) * x
        # no inequality constraints in this example
        cineq = np.zeros((0, 1))
        dcineq = np.zeros((nvars,0))
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
        # scale penalty if equality constraint violated and not improving
        #if np.abs(ceq) <
        if (np.abs(ceq) > eps_eq) and (np.abs(ceq) > np.abs(ceqold)):
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
        change = np.abs(x - xold).max()
        #
        if verbose:
            print("it.: {0}, obj.: {1:.10f}, ceq.: {2:.10f}, "
                  "rho: {3:.4e}, ch.: {4:.10f}".format(
                    i + 1,
                    obj,
                    ceq[0],
                    rho,
                    change))
        #
        if (change <= 1e-7):
            break
    print("Augmented Lagrangian Method")
    print("final objective: ", obj)
    print("final x: ", x[:, 0])
    print("final gradient: ", fgrad[:, 0])
    print("constraint vs ideal: ", (x**2).sum(), nvars)
    print("final lambda: ", float(lam[0, 0]))
    print("after {0} iterations".format(i + 1))

    return x, fgrad, lam, i + 1

def demonstrate_gcmma(nvars=3,
                      eqconstr_version=2,
                      verbose=False,
                      maxiter=2000,
                      maxiter_inner=15,
                      start_constrained=False):
    """
    Simple demonstration code for the use of the method of moving asymptotes
    minimizing the rosenbrock function in the interval [-1.5,1.5] with a 
    constraint that forces the sum of the squared variables equals the number 
    of variables. GCMMA and MMA in its original formulation do not treat 
    equality constraints.
    

    Parameters
    ----------
    nvars : float
        number of variables.
    eqconstr_version : int
        triggers two different versions of enforcing the equality constraint.
    verbose : bool
        whether to print out intermediate information.
    maxiter : int 
        maximum number of outer iterations.
    maxiter_inner : int
        maximum number of inner iterations.
    start_constrained : bool 
        if True, start from random initial guess that fulfills constraint.

    Returns
    -------
    None
    """
    if eqconstr_version == 1:
        n_constr = 1
    elif eqconstr_version == 2:
        n_constr = 2
        eps_eqconstr = 1e-5
    else:
        raise ValueError("only two versions (1,2) of the equality constraint exist.")
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars,1)
    xhist = [None,None]
    #x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = gcmma_defaultkws(x.shape[0],
                                    ft=None,
                                    n_constr=n_constr)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.5)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.5
    optimizer_kw["move"] = 1e-2
    #optimizer_kw["raa0"] = np.array([ [ optimizer_kw["raa0"] ]])
    kkttol = 0
    innerit_max = 15
    #
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
        # calculate objective function
        obj = rosen(x)
        dobj[:,:] = rosen_der(x)
        # calculate sensitivities
        # version 1 of equality constraint by a squared constraint
        if eqconstr_version == 1:
            constrs = ((x**2).sum()/nvars - 1)**2
            dconstr = 2*x /nvars * ((x**2).sum()/nvars - 1)
        # version 2 of equality constraint: use two inequalities
        elif eqconstr_version == 2:
            constrs = np.vstack( ((x**2).sum()/nvars - 1,
                                  1 - (x**2).sum()/nvars) ) + eps_eqconstr
            dconstr = np.column_stack( (2*x /nvars,- 2*x /nvars))
        #
        xval = x.copy()
        # update asymptotes
        optimizer_kw["low"], optimizer_kw["upp"], optimizer_kw["raa0"], optimizer_kw["raa"] = \
             asymp(outeriter=i, n=x.shape[0],
                   xval=x,
                   xold1=xhist[-1],
                   xold2=xhist[-2],
                   df0dx=dobj,
                   dfdx=dconstr.T,
                   **optimizer_kw)
        #
        xmma, ymma, zmma, lam, xsi, eta_mma, mu, zet, s, f0app, fapp = gcmmasub(
                                                     m=optimizer_kw["nconstr"],
                                                     n=x.shape[0],
                                                     iter=i,
                                                     xval=x,
                                                     xold1=xhist[-1],
                                                     xold2=xhist[-2],
                                                     f0val=obj,
                                                     df0dx=dobj,
                                                     fval=constrs,
                                                     dfdx=dconstr.T,
                                                     **optimizer_kw)
        # residual vector of the KKT conditions
        residu, kktnorm, residumax = kktcheck(m=optimizer_kw["nconstr"],
                                              n=x.shape[0],
                                              x=xmma,
                                              y=ymma,
                                              z=zmma,
                                              lam=lam,
                                              xsi=xsi,
                                              eta=eta_mma,
                                              mu=mu,
                                              zet=zet,
                                              s=s,
                                              df0dx=dobj,
                                              fval=constrs,
                                              dfdx=dconstr.T,
                                              **optimizer_kw)
        if kktnorm > kkttol:
            # recompute objective function and constraint function, but no
            # derivatives
            obj_new = rosen(xmma)
            if eqconstr_version == 1:
                constrs_new = np.array([ ((xmma**2).sum()/nvars - 1)**2 ])
            elif eqconstr_version == 2:
                constrs_new = np.vstack( ((xmma**2).sum()/nvars - 1,
                                        1 - (xmma**2).sum()/nvars) ) + eps_eqconstr
            # conservative check: approximating objective and constraint
            # functions become greater than or equal to the original functions
            conserv = concheck(m=optimizer_kw["nconstr"],
                               f0app=f0app,
                               f0valnew=obj_new,
                               fapp=fapp,
                               fvalnew=constrs_new,
                               **optimizer_kw)
            innerit = 0
            if conserv==0:
                # inner iteration
                for innerit in np.arange(innerit_max):
                    # update raa0 and raa:
                    optimizer_kw["raa0"], optimizer_kw["raa"] = raaupdate(
                                          xmma=xmma,
                                          xval=x,
                                          f0valnew=obj_new[0],
                                          fvalnew=constrs_new,#[:,None],
                                          f0app=f0app, fapp=fapp,
                                          **optimizer_kw)

                    # gcmma iteration with new raa0 and raa:
                    xmma, ymma, zmma, lam, xsi, eta_mma, mu, zet, s, f0app, fapp = gcmmasub(
                                                                 m=optimizer_kw["nconstr"],
                                                                 n=x.shape[0],
                                                                 iter=i,
                                                                 xval=x,
                                                                 xold1=xhist[-1],
                                                                 xold2=xhist[-2],
                                                                 f0val=obj,
                                                                 df0dx=dobj,
                                                                 fval=constrs,
                                                                 dfdx=dconstr.T,
                                                                 **optimizer_kw)
                    # recompute objective function and constraint function, but no
                    # derivatives
                    obj_new = rosen(xmma)
                    if eqconstr_version == 1:
                        constrs_new = np.array([ ((xmma**2).sum()/nvars - 1)**2 ])
                    elif eqconstr_version == 2:
                        constrs_new = np.vstack( ((xmma**2).sum()/nvars - 1,
                                                1 - (xmma**2).sum()/nvars) ) + eps_eqconstr
                    # check conservative (constraints are fulfilled)
                    conserv = concheck(m=optimizer_kw["nconstr"],
                                       f0app=f0app,
                                       f0valnew=obj_new,
                                       fapp=fapp,
                                       fvalnew=constrs_new,
                                       **optimizer_kw)
                    if conserv==1:
                        if verbose==2:
                            print("inner iteration finished after: ",innerit)
                        break
            # update x
            x[:] = xmma.copy()
            # delete oldest element of iteration history
            xhist.pop(0)
            xhist.append(xval)
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, constr.: {2:.10f}, ch.: {3:.10f}, kktnorm.: {4:.10f}".format(
                  int(i+1), obj[0], (x**2).sum()-nvars , change, kktnorm))
        if change <= 1e-7:
            break
    print("Globally Convergent MMA")
    print("final x: ", x[:,0])
    print("final gradient: ", dobj[:,0])
    print("constraint vs ideal: ", (x**2).sum(), nvars )
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

def demonstrate_mma(nvars=3,
                    eqconstr_version=1,
                    verbose=False,
                    maxiter=2000,
                    start_constrained=True):
    """
    Simple demonstration code for the use of the method of moving asymptotes
    minimizing the rosenbrock function in the interval [-1.5,1.5].

    Parameters
    ----------
    nvars : float
        number of variables.

    Returns
    -------
    None
    """
    if eqconstr_version == 1:
        n_constr = 1
    elif eqconstr_version == 2:
        n_constr = 2
        eps_eqconstr = 1e-5
    else:
        raise ValueError("only two versions (1,2) of the equality constraint exist.")
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars,1)
    xhist = [None,None]
    #x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=n_constr)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.5)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.5
    optimizer_kw["move"] = 1e-4
    #
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
        # calculate objective function
        obj = rosen(x)
        dobj[:,:] = rosen_der(x)
        # calculate sensitivities
        # version 1 of equality constraint by a squared constraint
        if eqconstr_version == 1:
            constrs = ((x**2).sum()/nvars - 1)**2
            dconstr = 2*x /nvars * ((x**2).sum()/nvars - 1)
        # version 2 of equality constraint: use two inequalities
        elif eqconstr_version == 2:
            constrs = np.vstack( ((x**2).sum()/nvars - 1,
                                  1 - (x**2).sum()/nvars) ) + eps_eqconstr
            dconstr = np.column_stack( (2*x /nvars,- 2*x /nvars))
        #
        xval = x.copy()
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"],
                                                             n=x.shape[0],
                                                             iter=i,
                                                             xval=x,
                                                             xold1=xhist[-1],
                                                             xold2=xhist[-2],
                                                             f0val=obj,
                                                             df0dx=dobj,
                                                             fval=constrs,
                                                             dfdx=dconstr.T,
                                                             **optimizer_kw)
        x = xmma.copy()
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, constr.: {2:.10f}, ch.: {3:.10f}".format(
                  int(i+1), obj[0], (x**2).sum()-nvars, change))
        if change <= 1e-7:
            break
    print("MMA")
    print("final x: ", x[:,0])
    print("final gradient: ", dobj[:,0])
    print("constraint vs ideal: ", (x**2).sum(), nvars )
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

if __name__ == "__main__":

    #
    verbose = True
    maxiter = int(1e2)
    #
    import sys
    if len(sys.argv)>1:
        verbose = bool(int(sys.argv[1]))
    if len(sys.argv)>2:
        maxiter = int(sys.argv[2])
    #
    demonstrate_alm_line(verbose=verbose, maxiter=maxiter)
    #demonstrate_mma(verbose=verbose, maxiter=maxiter)
    #demonstrate_gcmma(verbose=verbose, maxiter=maxiter)
