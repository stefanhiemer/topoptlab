import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.gradient_descent import barzilai_borwein,gradient_descent
from topoptlab.optimizer.mma_utils import mma_defaultkws,update_mma
from topoptlab.accelerators import anderson,diis 

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
    x = x / np.sqrt((x**2).sum()) 
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=0)
    # lower and upper bound for densities
    optimizer_kw["xmin"] = np.ones((nvars,1))*(-1.5)
    optimizer_kw["xmax"] = np.ones((nvars,1))*1.5
    # 
    dobj = np.zeros(x.shape)
    #
    print((x**2).sum())
    for i in np.arange(maxiter):
        #print(x)
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        constrs = np.array([(x**2).sum()-nvars])
        dconstr = 2*x / nvars
        #
        xval = x.copy()[None].T
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = update_mma(x=x,
                                                            xold1=xhist[-1],
                                                            xold2=xhist[-2],
                                                            xPhys=x,
                                                            obj=obj,
                                                            dobj=dobj,
                                                            constrs=constrs,
                                                            dconstr=dconstr,
                                                            iteration=i,
                                                            **optimizer_kw)
        x = xmma.copy().flatten()
        # update asymptotes 
        #optimizer_kw["low"] = np.maximum(x[:,None]-0.1,optimizer_kw["xmin"])
        #optimizer_kw["upp"] = np.minimum(x[:,None]+0.1,optimizer_kw["xmax"])
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                  i+1, obj, change))
            print((x**2).sum())
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
    demonstrate_mma(verbose=verbose, maxiter=maxiter)