# SPDX-License-Identifier: GPL-3.0-or-later
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.sparse import load_npz

from topoptlab.amg import standard_coarsening
from topoptlab.gmg import create_coarse_mask
from topoptlab.blocksparse.make_blocks import create_equal_blocks
from topoptlab.utils import map_eltoimg
from topoptlab.example_bc.lin_elast import mbb_2d
from topoptlab.example_bc.heat_conduction import heatplate_2d

def demo_mbb2d():
    return 60,20,None,2.4,2,0.5,"mbb2d"

def demo_heatplate2d():
    return 40,40,None,1.2,1,0.4,"heatplate2d"

if __name__ == "__main__":
    #
    nelx,nely,nelz,rmin,ndof,volfrac,name = demo_heatplate2d()
    #
    it = 25
    #
    mapping = partial(map_eltoimg,
                      nelx=nelx,nely=nely)
    #
    xPhys = np.loadtxt(f"examples/linsolvers/xPhys_{name}_{nelx}x{nely}-{rmin}-{volfrac}-0_{it}.csv",
                       delimiter=",")
    #
    x,y = np.meshgrid(np.linspace(0,1,nelx+1),
                      np.linspace(0,1,nely+1))
    x,y = np.repeat(x.flatten("F"),ndof),np.repeat(y.flatten("F"),ndof)
    #
    x = x*nelx - 0.5
    y = y*nely - 0.5
    # create laplacian
    L = load_npz(f"examples/linsolvers/stiffness-matrix_{name}_{nelx}x{nely}-{rmin}-{volfrac}-0_{it}.npz")
    #
    if ndof == 2 and nelz is None:
        u,f,fixed,free,springs = mbb_2d(nelx=nelx,nely=nely,nelz=nelz,
                                        ndof=ndof)
    elif ndof == 1 and nelz is None:
        u,f,fixed,free,springs = heatplate_2d(nelx=nelx,nely=nely,nelz=nelz,
                                              ndof=ndof)
    x = np.delete(x, fixed)
    y = np.delete(y, fixed)
    # goemetric multigrid
    gmg_mask = create_coarse_mask(nelx=nelx, 
                                  nely=nely,
                                  ndof=ndof, 
                                  stride=(2,2) )
    # delete fixed dofs
    gmg_mask = np.delete(gmg_mask, fixed)
    # algebraic multigrid
    amg_mask = standard_coarsening(A=L)
    # block_precond
    block_masks = create_equal_blocks(A=L,nblocks=2)
    #
    fig,axs = plt.subplots(1,3)
    #
    for i in range(3):
        axs[i].imshow(mapping(-xPhys), cmap='gray',
                       interpolation='none', norm=Normalize(vmin=-1, vmax=0))
        axs[i].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       labelbottom=False,
                       labelleft=False)
        axs[i].axis("off")
    #
    axs[0].scatter(x[gmg_mask],y[gmg_mask],c="r",s=0.5)
    axs[0].scatter(x[~gmg_mask],y[~gmg_mask],c="b",s=0.5)
    axs[0].axis("off")
    #
    axs[1].scatter(x[amg_mask],y[amg_mask],c="r",s=0.5)
    axs[1].scatter(x[~amg_mask],y[~amg_mask],c="b",s=0.5)
    axs[1].axis("off")
    #
    for mask,c in zip(block_masks,["r","b","g"]):
        axs[2].scatter(x[mask],y[mask],c=c,s=0.5)
    axs[2].axis("off")
    #
    plt.show()