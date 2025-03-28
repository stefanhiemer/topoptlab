import numpy as np
from scipy.optimize import rosen, rosen_der

from topoptlab.optimizer.gradient_descent import barzilai_borwein

def demonstrate_barzilai_borwein(nvars=3):
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
        #print(x_old)
        #print(xhist)
        #
        x = barzilai_borwein(x=x, dobj=dobj, 
                             xold=xhist[0], dobjold=dobjold,
                             xmin=-1., xmax=1., 
                             mode = "long",
                             el_flags=None, move=0.1)
        dobjold[:] = dobj
        #
        change = np.abs(x - xhist[-1]).max()
        #
        print("it.: {0} obj.: {1:.10f}, ch.: {2:.10f}".format(
                     i+1, obj, change))
        if change <= 1e-5:
            break
    print(x,dobj)
    return

if __name__ == "__main__":
    
    demonstrate_barzilai_borwein()