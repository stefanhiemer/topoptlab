from functools import partial

import numpy as np
from numpy import ones,asarray
from numpy.random import seed,rand
from scipy.ndimage import grey_closing,grey_opening,grey_dilation,grey_erosion
import matplotlib.pyplot as plt

from topoptlab.filters import assemble_convolution_filter,assemble_matrix_filter
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel
from topoptlab.geometries import sphere
from topoptlab.output_designs import threshold

def strel(radius, fill_value=1):
    """
    Create element flags for a sphere located at center with specified radius.

    Parameters
    ----------
    geo : str
        name of geometry.
    radius : float
        sphere radius.
    fill_value: int
        value that is prescribed to elements within sphere.

    Returns
    -------
    el_flags : np.ndarray
        element flags of shape (nelx*nely)

    """
    l = 1+int(2*r)
    center = np.median([0,l-1])
    center = (center,center)
    el = np.arange(int(l**2))
    i = np.floor(el/l)
    j = el%l
    mask = (i-center[0])**2 + (j-center[1])**2 <= radius**2 #nely/3

    #
    el_flags = np.zeros(el.shape)
    el_flags[mask] = fill_value
    return el_flags.reshape(l,l)

def diracdelta(nelx,nely):
    # densities
    x = sphere(nelx,nely,
               ((nelx-1)/2,(nely-1)/2),
               radius=1,fill_value=1.)
    return x

def display(x,nelx,nely,r):
    #
    x = map_eltoimg(quant=x, nelx=nelx, nely=nely)
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
    solidsafe = grey_opening(x,size=structure_R.shape,
                               structure=structure_R,
                               mode="nearest",cval=0.)
    voidsafe = (1-x) - voidviolation
    # erosion countermeasure
    voiderosion = x*grey_dilation(voidviolation,size=structure.shape,
                                  structure=structure,
                                  mode="nearest",cval=0.)
    # dilation in counter measures
    soliddilation = (1-x)*grey_dilation(solidviolation,size=structure.shape,
                                        structure=structure,
                                        mode="nearest",cval=0.)
    #
    fig,ax = plt.subplots(2,2)
    # img with violations highlighted
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidviolation == 1] = [1, 0, 0]
    img[voidviolation == 1] = [0, 1, 0]
    ax[0,0].imshow(img)
    # img with "safe" regions
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidsafe == 1] = [1, 0, 0]
    img[voidsafe == 1] = [0, 1, 0]
    ax[0,1].imshow(img)
    # img with erosion
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[solidviolation == 1] = [1, 1, 1]
    img[voiderosion-1 == 1] = [1, 1, 1]
    ax[1,0].imshow(img)
    # img with "safe" regions
    img = np.ones(x.shape + tuple([3]))
    img[x==1] = [0,0,0]
    img[soliddilation-1 == 1] = [0, 0, 1]
    img[voidviolation == 1] = [0, 0, 0]
    ax[1,1].imshow(img)
    # img with counter measures
    for i in range(4):
        row,col = int(i%2),int(np.floor(i/2))
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
