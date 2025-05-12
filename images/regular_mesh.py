import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D

from roman import toRoman

from topoptlab.utils import map_eltoimg,map_eltovoxel

def create_nrs(nelx,nely,nelz=None):
    if nelz is None:
        return map_eltoimg(np.arange(nelx*nely),nelx=nelx,nely=nely)
    else:
        return map_eltovoxel(np.arange(nelx*nely*nelz),
                             nelx=nelx,nely=nely,nelz=nelz)

def plot_meshnumbering2d(nelx=4,nely=3,ndof=2,
                         eloffset=0,ndoffset=0,
                         ax=None):
    # create element and node numbering
    elgrid = create_nrs(nelx,nely)
    ndgrid = create_nrs(nelx+1,(nely+1)*ndof)
    # 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        show=True
    else:
        show=False
    # add cell numbers
    for i in range(nely):
        for j in range(nelx):
            ax.text(x=j, y=i, s=toRoman(int(eloffset+elgrid[i,j])),
                    ha='center', va='center', fontsize=12)
    # add node numbers
    for i in range(nely+1):
        for j in range(nelx+1):
            ax.text(x=j-0.3, y=i-0.4, 
                    s=",".join([str(s+ndoffset) for s in \
                                ndgrid[i*ndof:(i+1)*ndof,j]]), 
                    ha='center', va='center', fontsize=12)
    # grid lines
    ax.set_xticks(np.arange(0.5, nelx+0.5, 1), minor=False)
    ax.set_yticks(np.arange(0.5, nely+0.5, 1), minor=False)
    ax.grid(which='major', color='black', linestyle='-', linewidth=1)
    # remove ticks on x and y axis
    ax.tick_params(axis='both',which='both',
                   bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    # set axis limits
    ax.set_xlim(-0.5, nelx-0.5)
    ax.set_ylim(nely-0.5, -0.5)
    if show:
        plt.show()
    return

def plot_meshnumbering3d(nelx=4,nely=3,nelz=2,ndof=2):
    # create element and node numbering
    elgrid = create_nrs(nelx,nely,nelz)
    ndgrid = create_nrs(nelx+1,(nely+1)*ndof,nelz)
    # 
    #fig = plt.figure(figsize=(12, 8))
    #gs = GridSpec(3, 2, width_ratios=[1, 1])
    #
    fig, axs = plt.subplots(ncols=2, nrows=nelz+1)
    gs = axs[1, 0].get_gridspec()
    # remove the underlying Axes
    for ax in axs[:, 0]:
        ax.remove()
    # First plot (takes up the entire left side)
    ax1 = fig.add_subplot(gs[:,0], projection='3d')
    # 3d plot element grid
    for i in range(nely):
        for j in range(nelx):
            for k in range(nelz):
                ax1.text(x=j, y=i, z = k,
                         s=toRoman(int(elgrid[k,i,j])), 
                         ha='center', va='center', fontsize=12)
    # add node numbers
    #for i in range(nely+1):
    #    for j in range(nelx+1):
    #        for k in range(nelz+1):
    #            ax1.text(x=j-0.3, y=i-0.4,z=k,
    #                     s=",".join([str(s) for s in \
    #                                 ndgrid[k,i*ndof:(i+1)*ndof,j]]), 
    #                     ha='center', va='center', fontsize=12)
    # grid lines
    ax1.set_xticks(np.arange(0.5, nelx+0.5, 1), minor=False)
    ax1.set_yticks(np.arange(0.5, nely+0.5, 1), minor=False)
    ax1.set_zticks(np.arange(0.5, nelz+0.5, 1), minor=False)
    ax1.grid(which='major', color='black', linestyle='-', linewidth=1)
    # remove ticks on x and y axis
    ax1.tick_params(axis='both',which='both',
                    bottom=False, labelbottom=False,
                    left=False, labelleft=False)
    # set axis limits
    ax1.set_xlim(-0.5, nelx-0.5)
    ax1.set_ylim(nely-0.5, -0.5)
    ax1.set_zlim(-0.5, nelz-0.5)
    # 2d plots 
    for i in range(nelz+1):
        axs[i,1].set_title("z = "+str(i))
        plot_meshnumbering2d(nelx=nelx,nely=nely,ndof=ndof,
                             eloffset=i*nelx*nely,ndoffset=i*(nelx+1)*(nely+1),
                             ax=axs[i,1])
    # Adjust layout to prevent overlap
    fig.tight_layout()
    plt.show()
    return
    
if __name__ == "__main__":
    #
    plot_meshnumbering3d(nelx=4,nely=3,ndof=2)
    
