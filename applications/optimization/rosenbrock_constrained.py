import warnings

import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.mma_utils import mma_defaultkws,gcmma_defaultkws
from mmapy import mmasub,gcmmasub, kktcheck, asymp, concheck, raaupdate

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
        if change <= 1e-9:
            break
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
        if change <= 1e-9:
            break
    print("final x: ", x[:,0])
    print("final gradient: ", dobj[:,0])
    print("constraint vs ideal: ", (x**2).sum(), nvars )
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

if __name__ == "__main__":

    #
    verbose = False
    maxiter = 1e3
    #
    import sys
    if len(sys.argv)>1:
        verbose = bool(int(sys.argv[1]))
    if len(sys.argv)>2:
        maxiter = int(sys.argv[2])
    #
    demonstrate_mma(verbose=verbose, maxiter=maxiter)
    demonstrate_gcmma(verbose=verbose, maxiter=maxiter)
