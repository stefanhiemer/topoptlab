from functools import partial

import numpy as np
from numpy import ones,asarray
from numpy.random import seed,rand
from scipy.ndimage import convolve

from topoptlab.filters import find_eta, eta_projection

if __name__ == "__main__":
    #
    n=10
    volfrac = 0.1
    # random densities
    np.random.seed(0)
    x = np.random.rand(n)
    #
    print("initial vol. frac.: ",x.mean())
    print("ideal vol. frac.: ",volfrac)
    #
    beta = 10
    #
    print("vol. frac. after filtering with eta=1: ",
          eta_projection(xTilde=x,eta=1.,beta=beta).mean() )
    print("vol. frac. after filtering with eta=0: ",
          eta_projection(xTilde=x,eta=0.,beta=beta).mean() )
    print("vol. frac. after filtering: ",
          eta_projection(xTilde=x,
                         eta=find_eta(xTilde=x,beta=beta,eta0=0.5,volfrac=volfrac),
                         beta=beta).mean() )
