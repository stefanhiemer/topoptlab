from functools import partial

import numpy as np

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# map element data to img/voxel
from topoptlab.utils import map_eltoimg,map_imgtoel,map_eltovoxel,map_voxeltoel


# The real main driver
if __name__ == "__main__":
    #
    nelx,nely,nelz = 4,3,2
    xPhys = np.arange(nelx*nely*nelz)/(nelx*nely*nelz)
    #
    mapping = partial(map_eltovoxel,
                      nelx=nelx,nely=nely,nelz=nelz)
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"})
    cmap = plt.get_cmap("gray_r")
    #im = ax.voxels(filled = mapping(np.ones(xPhys.shape,dtype=bool)),
    #               facecolors = mapping(cmap(xPhys)))
    #plotfunc = im[0].set_facecolors
    ax.plot_surface(x, y, z)
    print(type(im))
    import sys 
    sys.exit()
    
    #plotfunc = im[0].set_facecolors
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    ax.axis("off")
    fig.show()
    
    for i in np.arange(200):
        #
        xPhys = np.random.rand(nelx*nely*nelz)
        #
        plotfunc(cmap(xPhys))
        fig.canvas.draw()
        plt.pause(0.01)
    #plt.show()
    input()