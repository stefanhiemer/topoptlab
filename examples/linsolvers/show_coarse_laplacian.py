# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LaplacianNd

from topoptlab.amg import standard_coarsening
from topoptlab.gmg import create_coarse_mask
from topoptlab.blocksparse_precond import create_primitive_blocks

if __name__ == "__main__":
    #
    grid_shape=(11,11)
    #
    x,y = np.meshgrid(np.linspace(0,1,11),
                      np.linspace(0,1,11))
    x,y = x.flatten("F"),y.flatten("F")
    # create laplacian
    L = LaplacianNd(grid_shape=grid_shape,
                    boundary_conditions = "neumann").tosparse().tocsc()
    L = L.astype(np.float64)
    print(L.shape)
    # goemetric multigrid
    gmg_mask = create_coarse_mask(nelx=grid_shape[0]-1, 
                                  nely=grid_shape[0]-1)
    print(gmg_mask.shape,x.shape,y.shape)
    # algebraic multigrid
    amg_mask = standard_coarsening(A=L)
    # block_precond
    block_masks = create_primitive_blocks(A=L,nblocks=2)
    #
    fig,axs = plt.subplots(1,3)
    #
    axs[0].scatter(x[gmg_mask],y[gmg_mask],c="r")
    axs[0].scatter(x[~gmg_mask],y[~gmg_mask],c="b")
    axs[0].axis("off")
    #
    axs[1].scatter(x[amg_mask],y[amg_mask],c="r")
    axs[1].scatter(x[~amg_mask],y[~amg_mask],c="b")
    axs[1].axis("off")
    #
    for mask,c in zip(block_masks,["r","b","g"]):
        axs[2].scatter(x[mask],y[mask],c=c)
    axs[2].axis("off")
    #
    plt.show()