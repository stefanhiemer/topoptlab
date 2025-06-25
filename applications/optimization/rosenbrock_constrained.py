import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.mma_utils import mma_defaultkws,gcmma_defaultkws 
from mmapy import mmasub,gcmmasub, kktcheck, asymp, concheck, raaupdate

def demonstrate_gcmma(nvars=3,
                      verbose=False,
                      maxiter=2000):
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
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars)
    xhist = [x.copy(),x.copy()]
    #x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = gcmma_defaultkws(x.shape[0],ft=None,n_constr=0)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.5)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.5
    optimizer_kw["move"] = 1e-4
    #
    if 1:
        return
    # 
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
        #print(x)
        # calculate objective function
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        # calculate sensitivities
        constrs = np.array([(x**2).sum()/nvars])
        dconstr = 2*x / nvars
        #
        
        #
        xval = x.copy()[None].T
        #
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                                                     m=optimizer_kw["nconstr"], 
                                                     n=x.shape[0],
                                                     iter=i,
                                                     xval=x[:,None],
                                                     xold1=xhist[-1],
                                                     xold2=xhist[-2],
                                                     f0val=obj,
                                                     df0dx=dobj[:,None],
                                                     fval=constrs[:,None],
                                                     dfdx=dconstr,
                                                     **optimizer_kw)
        # residual vector of the KKT conditions
        residu, kktnorm, residumax = kktcheck(m=optimizer_kw["nconstr"], 
                                              n=x.shape[0],
                                              xmma=xmma, 
                                              ymma=ymma, 
                                              zmma=zmma, 
                                              lam=lam, 
                                              xsi=xsi, 
                                              eta=eta, 
                                              mu=mu, 
                                              zet=zet, 
                                              s=s,
                                              df0dx=dobj[:,None], 
                                              fval=constrs[:,None],
                                              dfdx=dconstr,
                                              **optimizer_kw)
        # recompute objective function and constraint function, but no 
        # derivatives
        obj_new = rosen(x)
        constrs_new = np.array([(x**2).sum()/nvars])
        # conservative check (True if constraints are fulfilled)
        conserv = concheck(m=optimizer_kw["nconstr"],
                           f0app=f0app, 
                           f0valnew=obj_new, 
                           fapp=fapp, 
                           fvalnew=constrs_new, 
                           **optimizer_kw)
        
        #optimizer_kw["low"], optimizer_kw["upp"], optimizer_kw["raa0"], optimizer_kw["raa"] \
        #    = asymp(outeriter=i, n=x.shape[0], xval=x[:,None], 
        #            xold1=xhist[-1], xold2=xhist[-2], 
        #            df0dx=dobj[:,None], dfdx=dconstr,
        #            **optimizer_kw)
        
        # update asymptotes 
        optimizer_kw["low"] = np.maximum(x[:,None]-optimizer_kw["move"],optimizer_kw["xmin"])
        optimizer_kw["upp"] = np.minimum(x[:,None]+optimizer_kw["move"],optimizer_kw["xmax"])
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                  i+1, obj, change))
            print((x**2).sum()-nvars)
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

def demonstrate_mma(nvars=3,
                    verbose=False,
                    maxiter=2000):
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
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars)
    xhist = [x.copy(),x.copy()]
    #x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=0)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.5)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.5
    optimizer_kw["move"] = 1e-4
    # 
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
        #print(x)
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        constrs = np.array([(x**2).sum()/nvars])
        dconstr = 2*x / nvars
        #
        xval = x.copy()[None].T
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = mmasub(m=optimizer_kw["nconstr"], 
                                                             n=x.shape[0],
                                                             iter=i,
                                                             xval=x[:,None],
                                                             xold1=xhist[-1],
                                                             xold2=xhist[-2],
                                                             f0val=obj,
                                                             df0dx=dobj[:,None],
                                                             fval=constrs[:,None],
                                                             dfdx=dconstr,
                                                             **optimizer_kw)
        x = xmma.copy().flatten()
        # update asymptotes 
        optimizer_kw["low"] = np.maximum(x[:,None]-optimizer_kw["move"],optimizer_kw["xmin"])
        optimizer_kw["upp"] = np.minimum(x[:,None]+optimizer_kw["move"],optimizer_kw["xmax"])
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                  i+1, obj, change))
            print((x**2).sum()-nvars)
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

if __name__ == "__main__":
    
    #
    verbose = True
    maxiter = 1e2
    #
    import sys
    if len(sys.argv)>1: 
        verbose = bool(int(sys.argv[1]))
    if len(sys.argv)>2: 
        maxiter = int(sys.argv[2])
    #
    demonstrate_gcmma(verbose=verbose, maxiter=maxiter)