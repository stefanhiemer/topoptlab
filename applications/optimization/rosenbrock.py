import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.gradient_descent import barzilai_borwein
from topoptlab.optimizer.mma_utils import mma_defaultkws,update_mma

def demonstrate_barzilai_borwein(nvars=3):
    """
    Simple demonstration code for the use of the barizilai borwein optimizer
    by minimizing the rosenbrock function in the interval [-1.5,1.5].
    
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
    dobjold = rosen_der(x)
    #
    #
    xhist = [None,
             x]
    x = x - dobjold * 1e-8
    # 
    dobj = np.zeros(dobjold.shape)
    #
    for i in np.arange(2000):
        #print(x)
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        xhist.pop(0)
        xhist.append(x[:])
        #
        x = barzilai_borwein(x=x, dobj=dobj, 
                             xold=xhist[0], dobjold=dobjold,
                             xmin=-1.5, xmax=1.5, 
                             step_mode = "long",
                             el_flags=None, move=0.1)
        dobjold[:] = dobj
        #
        change = np.abs(x - xhist[-1]).max()
        #
        print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    return

def demonstrate_mma(nvars=3):
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
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=0)
    # 
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(2000):
        #print(x)
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        constrs = np.array([(x**2).sum()])
        dconstr = 2*x
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
        optimizer_kw["low"] = low
        optimizer_kw["upp"] = upp
        # delete oldest element of iteration history
        xhist.pop(0)
        xhist.append(xval)
        #
        change = np.abs(x - xhist[-1]).max()
        #
        print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    return

if __name__ == "__main__":
    
    #demonstrate_barzilai_borwein()
    demonstrate_mma()