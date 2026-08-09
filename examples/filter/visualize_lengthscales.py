# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial

import numpy as np
from numpy import ones,asarray
from numpy.random import seed,rand
from scipy.ndimage import grey_dilation,gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.utils import map_eltoimg
from topoptlab.geometries import sphere
from topoptlab.output_designs import threshold
from topoptlab.design_analysis import lengthscale_violations

def display(x,nelx,nely,r):
    #
    r = int(r)
    l = 1+int(2*r)
    structure = sphere(l,l,
                       (np.median([0,l-1]),np.median([0,l-1])),
                       radius=r,fill_value=1.)
    structure = map_eltoimg(structure, l, l)
    solidviolation, voidviolation = lengthscale_violations(x=x,
                                                           nelx=nelx,
                                                           nely=nely,
                                                           r=r,
                                                           nelz=nelz)
    #
    x = map_eltoimg(quant=x, 
                    nelx=nelx, 
                    nely=nely)
    #
    fig,ax = plt.subplots(1,2)
    # img with violations highlighted
    ax[0].imshow(-x, 
                 cmap="gray", 
                 interpolation='none',
                 norm=Normalize(vmin=-1, vmax=0))
    ax[0].set_title("original")
    # img with "safe" regions
    img = np.ones(x.shape[:-1] + tuple([3]))
    img[x[...,0]==1,:] = 0
    img[solidviolation[...,0]==1,0] = 1
    img[voidviolation[...,0]==1,0] = 0
    img[voidviolation[...,0]==1,1] = 0
    ax[1].imshow(img,
                 cmap='viridis', 
                 interpolation='none',
                 norm=Normalize(vmin=-1, vmax=0))
    ax[1].set_title("original with highlighted violations")
    # img with counter measures
    for col in range(2):
        ax[col].tick_params(axis='both',
                            which='both',
                            bottom=False,
                            left=False,
                            labelbottom=False,
                            labelleft=False)
    plt.show()
    return

if __name__ == "__main__":
    #
    nelx = 240
    nely = 80
    nelz = None
    r = 4.
    #x = diracdelta(nelx=nelx,nely=nely)
    x = np.loadtxt("mbb2d_240x80_v0.5_ft0.csv",delimiter=",")[:,None]
    x = threshold(xPhys=x,volfrac=0.5)
    display(x=x,nelx=nelx,nely=nely,r=r)
