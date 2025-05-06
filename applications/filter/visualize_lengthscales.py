from functools import partial

import numpy as np
from numpy import ones,asarray
from numpy.random import seed,rand
from scipy.ndimage import grey_dilation,grey_erosion,gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
from topoptlab.geometries import sphere, diracdelta
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
    #
    #solidviolation = x-grey_opening(x,size=structure.shape,
    #                                structure=structure,
    #                                mode="nearest",cval=0.)
    #voidviolation = grey_closing(x,size=structure.shape,
    #                             structure=structure,
    #                             mode="nearest",cval=0.)-x
    solidviolation, voidviolation = lengthscale_violations(x=x,
                                                           nelx=nelx,
                                                           nely=nely,
                                                           r=r,
                                                           nelz=nelz)
    #
    R = int(2*r)
    L = 1+int(2*R)
    structure_R = sphere(L,L,
                       (np.median([0,L-1]),np.median([0,L-1])),
                       radius=R,fill_value=1.)
    structure_R = map_eltoimg(structure_R, L, L)
    #
    _solidviolation, _voidviolation = lengthscale_violations(x=x,
                                                           nelx=nelx,
                                                           nely=nely,
                                                           r=R,
                                                           nelz=nelz)
    #
    x = map_eltoimg(quant=x, nelx=nelx, nely=nely)
    #
    solidsafe = x - _solidviolation
    voidsafe = (1-x) - _voidviolation
    # erosion countermeasure
    voiderosion = x*grey_dilation(voidviolation,size=structure.shape,
                                  structure=structure,
                                  mode="nearest",cval=0.)
    # dilation in counter measures
    soliddilation = (1-x)*grey_dilation(solidviolation,size=structure.shape,
                                        structure=structure,
                                        mode="nearest",cval=0.)
    #
    xnew = gaussian_filter(x,sigma=r, mode='nearest')
    #
    fig,ax = plt.subplots(3,2)
    # img with violations highlighted
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidviolation == 1] = [1, 0, 0]
    img[voidviolation == 1] = [0, 1, 0]
    ax[0,0].imshow(img)
    ax[0,0].set_title("original with highlighted violations")
    # img with "safe" regions
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidsafe == 1] = [1, 0, 0]
    img[voidsafe == 1] = [0, 1, 0]
    ax[0,1].imshow(img)
    ax[0,1].set_title("safe regions")
    # img with erosion
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidviolation == 1] = [1, 1, 1]
    img[voiderosion-1 == 1] = [1, 1, 1]
    ax[1,0].imshow(img)
    ax[1,0].set_title("Erosion")
    # img with "safe" regions
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[soliddilation-1 == 1] = [1, 0, 0]
    img[voidviolation == 1] = [0, 1, 0]
    ax[1,1].imshow(img)
    ax[1,1].set_title("Dilation")
    #
    img = np.ones(x.shape + tuple([3]))
    print(xnew)
    ax[2,0].imshow(1-xnew, cmap='gray')
    ax[2,0].set_title("Countermeasures")
    # img with counter measures
    for i in range(6):
        row,col = int(i%3),int(np.floor(i/3))
        ax[row,col].tick_params(axis='both',
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
    x = np.loadtxt("mbb2d_240x80_v0.5_ft0.csv",delimiter=",")
    x = threshold(xPhys=x,volfrac=0.5)
    display(x=x,nelx=nelx,nely=nely,r=r)
