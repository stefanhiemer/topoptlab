import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.gradient_descent import barzilai_borwein,gradient_descent
from topoptlab.optimizer.mma_utils import mma_defaultkws,update_mma
from topoptlab.accelerators import anderson,diis

def demonstrate_diis(nvars=3,q=5,q0=20,
                     max_history=5,
                     damp=0.9,
                     mix=0.9,
                     verbose=False,
                     maxiter=2000):
    """
    Simple demonstration code for the use of the periodic anderson acceleration 
    as optimizer by minimizing the rosenbrock function in the interval [-1.5,1.5].
    
    Parameters
    ----------
    nvars : int
        number of variables.
    accel_freq : int
        acceleration frequency.
    accel_start : int
        first iteration to start using acceleration.
    max_history : int
        maximum number of past results used for acceleration.
    damp : float
        damping of the DIIS update.
    mix : float
        mixing if not DIIS update is done.
    verbose : bool
        if True, information about the iteration is printed.
    
    Returns
    -------
    x : np.ndarray shape (nvars)
        final iterate.
    dobj : np.ndarray (nvars)
        final gradient.
    it : int
        number of iterations until final result.
    """
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars)
    dobjold = rosen_der(x)
    #
    xhist = [x]
    x = x - dobjold * 1e-8
    # 
    dobj = np.zeros(dobjold.shape)
    #
    for i in np.arange(maxiter):
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        x = gradient_descent(x=x, dobj=dobj, 
                             stepsize=1.75e-3,
                             xmin=-1.5, xmax=1.5,
                             el_flags=None, move=0.1)
        dobjold[:] = dobj
        #
        if ((i-q0) % q) == 0 and i >= q0:
            x = diis(x=x,xhist=xhist,
                     max_history=max_history,
                     damp=damp)
        else:
            x = xhist[-1]*(1-mix) + x*mix
        #
        xhist.append(x)
        change = np.abs(x - xhist[-2]).max()
        if len(xhist)> max_history+1:
            xhist = xhist[-q-1:]
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return

def demonstrate_anderson(nvars=3,accel_freq=5,accel_start=20,
                         max_history=5,
                         damp=0.9,
                         mix=0.9,
                         verbose=False,
                         maxiter=2000):
    """
    Simple demonstration code for the use of the periodic anderson acceleration 
    as optimizer by minimizing the rosenbrock function in the interval [-1.5,1.5].
    
    Parameters
    ----------
    nvars : int
        number of variables.
    accel_freq : int
        acceleration frequency.
    accel_start : int
        first iteration to start using acceleration.
    max_history : int
        maximum number of past results used for acceleration.
    damp : float
        damping of the Anderson update.
    mix : float
        mixing if not Anderson update is done.
    verbose : bool
        if True, information about the iteration is printed.
    
    Returns
    -------
    x : np.ndarray shape (nvars)
        final iterate.
    dobj : np.ndarray (nvars)
        final gradient.
    it : int
        number of iterations until final result.
    """
    #
    print("gradient descent with Anderson acceleration")
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars)
    dobjold = rosen_der(x)
    #
    xhist = [x]
    x = x - dobjold * 1e-8
    # 
    dobj = np.zeros(dobjold.shape)
    #
    for i in np.arange(1e5):
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        x = gradient_descent(x=x, dobj=dobj, 
                             stepsize=1.75e-3,
                             xmin=-1.5, xmax=1.5,
                             el_flags=None, move=0.1)
        dobjold[:] = dobj
        #
        if ((i-accel_start) % accel_freq) == 0 and i >= accel_start:
            x = anderson(x=x,xhist=xhist,
                         max_history=max_history,
                         damp=damp)
        else:
            x = xhist[-1]*(1-mix) + x*mix
        #
        xhist.append(x)
        change = np.abs(x - xhist[-2]).max()
        if len(xhist)> max_history+1:
            xhist = xhist[-max_history-1:]
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

def demonstrate_gradient_descent(nvars=3,
                                 verbose=False,
                                 maxiter=2000):
    """
    Simple demonstration code for constrained gradient descent 
    optimizer by minimizing the rosenbrock function in the interval [-1.5,1.5].
    
    Parameters
    ----------
    nvars : float
        number of variables.
    
    Returns
    -------
    None
    """
    #
    print("gradient descent")
    #
    q = 1
    #
    np.random.seed(1)
    #
    x = np.random.rand(nvars)
    dobjold = rosen_der(x)
    #
    xhist = [x]
    x = x - dobjold * 1e-8
    # 
    dobj = np.zeros(dobjold.shape)
    #
    for i in np.arange(1e5):
        #
        obj = rosen(x)
        dobj[:] = rosen_der(x)
        #
        x = gradient_descent(x=x, dobj=dobj, 
                             stepsize=1.425e-3,
                             xmin=-1.5, xmax=1.5,
                             el_flags=None, move=0.1)
        dobjold[:] = dobj
        #
        xhist.append(x)
        change = np.abs(x - xhist[-2]).max()
        if len(xhist)> q+1:
            xhist = xhist[-q-1:]
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

def demonstrate_barzilai_borwein(nvars=3,
                                 verbose=False,
                                 maxiter=2000):
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
    print("barzilai_borwein")
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
    for i in np.arange(maxiter):
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
                             el_flags=None, move=1e-1)
        dobjold[:] = dobj
        #
        change = np.abs(x - xhist[-1]).max()
        #
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                  i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return

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
    #
    optimizer_kw = mma_defaultkws(x.shape[0],ft=None,n_constr=0)
    # 
    dobj = np.zeros(x.shape)
    #
    for i in np.arange(maxiter):
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
        if verbose:
            print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                  i+1, obj, change))
        if change <= 1e-9:
            break
    print("final x: ", x)
    print("final gradient: ", dobj)
    print("after {0} iterations".format(int(i+1)))
    return x, dobj,i+1

if __name__ == "__main__":
    
    #
    verbose = False
    #
    import sys
    if len(sys.argv)>1: 
        verbose = bool(int(sys.argv[1]))
    if len(sys.argv)>2: 
        maxiter = int(sys.argv[2])
    #
    demonstrate_barzilai_borwein(verbose=verbose, maxiter=maxiter)
    demonstrate_mma(verbose=verbose, maxiter=maxiter)
    demonstrate_gradient_descent(verbose=verbose, maxiter=maxiter)
    demonstrate_anderson(mix=1.,damp=1.,verbose=verbose, maxiter=maxiter)
    demonstrate_diis(mix=0.5,damp=0.5,verbose=verbose, maxiter=maxiter)