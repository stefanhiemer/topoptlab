import warnings

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
    xhist = [None,None]
    x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = gcmma_defaultkws(x.shape[0],ft=None,n_constr=1)
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
        #print(x)
        # calculate objective function
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        # calculate sensitivities
        constrs = np.array([ ((x**2).sum()/nvars - 1)**2 ])
        dconstr = 2*x /nvars * ((x**2).sum()/nvars - 1) 
        #
        xval = x.copy()[None].T
        # update asymptotes 
        optimizer_kw["low"], optimizer_kw["upp"], optimizer_kw["raa0"], optimizer_kw["raa"] = \
             asymp(outeriter=i, n=x.shape[0], 
                   xval=x[:,None], 
                   xold1=xhist[-1],
                   xold2=xhist[-2],
                   df0dx=dobj[:,None], 
                   dfdx=dconstr[None,:],
                   **optimizer_kw)
        #
        """
        print("epsimin ", np.shape(optimizer_kw["epsimin"]))
        print("xval, xmin, xmax ",x[:,None].shape, 
                                  optimizer_kw["xmin"].shape, 
                                  optimizer_kw["xmax"].shape)
        print("low,upp", optimizer_kw["low"].shape ,optimizer_kw["upp"].shape)
        print("raa0, raa",optimizer_kw["raa0"].shape, optimizer_kw["raa"].shape)
        print("f0val, df0dx, fval, dfdx", np.shape(obj), 
                                          dobj[:,None].shape, 
                                          np.shape(constrs[0]), dconstr[None,:].shape)
        print("a,a0,c,d ",np.shape(optimizer_kw["a"]),
                          np.shape(optimizer_kw["a"]),
                          np.shape(optimizer_kw["c"]),
                          np.shape(optimizer_kw["d"]))
        """
        #
        xmma, ymma, zmma, lam, xsi, eta_mma, mu, zet, s, f0app, fapp = gcmmasub(
                                                     m=optimizer_kw["nconstr"], 
                                                     n=x.shape[0],
                                                     iter=i,
                                                     xval=x[:,None],
                                                     xold1=xhist[-1],
                                                     xold2=xhist[-2],
                                                     f0val=obj,
                                                     df0dx=dobj[:,None],
                                                     fval=constrs[0],
                                                     dfdx=dconstr[None,:],
                                                     **optimizer_kw)
        #import sys 
        #sys.exit()
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
                                              df0dx=dobj[:,None], 
                                              fval=constrs[:,None],
                                              dfdx=dconstr[None,:],
                                              **optimizer_kw)
        if kktnorm > kkttol:
            # recompute objective function and constraint function, but no 
            # derivatives
            obj_new = rosen(xmma[:,0])
            constrs_new = np.array([ ((xmma[:,0]**2).sum()/nvars - 1)**2 ])
            # conservative check: approximating objective and constraint 
            # functions become greater than or equal to the original functions
            conserv = concheck(m=optimizer_kw["nconstr"],
                               f0app=f0app, 
                               f0valnew=obj_new, 
                               fapp=fapp, 
                               fvalnew=constrs_new[:,None],
                               **optimizer_kw)
            #print(conserv)
            #if i==2:
            #    import sys 
            #    sys.exit()
            innerit = 0
            if conserv==0:
                # inner iteration
                for innerit in np.arange(innerit_max):
                    # update raa0 and raa:
                    optimizer_kw["raa0"], optimizer_kw["raa"] = raaupdate(
                                          xmma=xmma, 
                                          xval=x[:,None],
                                          f0valnew=obj_new, 
                                          fvalnew=constrs_new[:,None], 
                                          f0app=f0app, fapp=fapp, 
                                          **optimizer_kw)
                    
                    # gcmma iteration with new raa0 and raa:
                    xmma, ymma, zmma, lam, xsi, eta_mma, mu, zet, s, f0app, fapp = gcmmasub(
                                                                 m=optimizer_kw["nconstr"], 
                                                                 n=x.shape[0],
                                                                 iter=i,
                                                                 xval=x[:,None],
                                                                 xold1=xhist[-1],
                                                                 xold2=xhist[-2],
                                                                 f0val=obj,
                                                                 df0dx=dobj[:,None],
                                                                 fval=constrs[0],
                                                                 dfdx=dconstr[None,:],
                                                                 **optimizer_kw)
                    # recompute objective function and constraint function, but no 
                    # derivatives
                    obj_new = rosen(xmma[:,0])
                    constrs_new = np.array([ ((xmma[:,0]**2).sum()/nvars - 1)**2 ])
                    # check conservative (constraints are fulfilled)
                    conserv = concheck(m=optimizer_kw["nconstr"],
                                       f0app=f0app, 
                                       f0valnew=obj_new, 
                                       fapp=fapp, 
                                       fvalnew=constrs_new[:,None], 
                                       **optimizer_kw)
                    if conserv==1:
                        print("inner iteration finished after: ",innerit)
                        break
            # update x
            x = xmma.copy().flatten()
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
                  i+1, obj, (x**2).sum()-nvars, change,kktnorm))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("constraint: ", (x**2).sum() )
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
    xhist = [None, None]
    x = x * np.sqrt( nvars/(x**2).sum())
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=1)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.
    optimizer_kw["move"] = 1e-3
    # 
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
        #print(x)
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        constrs = np.array([(x**2).sum() - nvars])
        dconstr = 2*x
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
            print("it.: {0} obj.: {1:.10f}, constr.: {2:.10f}, ch.: {3:.10f}".format(
                  i+1, obj, (x**2).sum()-nvars, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("constraint: ", (x**2).sum() )
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

if __name__ == "__main__":
    
    #
    verbose = True
    maxiter = 21
    #
    import sys
    if len(sys.argv)>1: 
        verbose = bool(int(sys.argv[1]))
    if len(sys.argv)>2: 
        maxiter = int(sys.argv[2])
    #
    demonstrate_gcmma(verbose=verbose, maxiter=maxiter)