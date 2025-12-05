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
        fig, ax = plt.subplots(figsize=(2.6, 2))
        show=True
    else:
        show=False
    # add cell numbers
    for i in range(nely):
        for j in range(nelx):
            ax.text(x=j, y=i, s=toRoman(int(eloffset+elgrid[i,j])),
                    ha='center', va='center', fontsize=10)
    # add node numbers
    for i in range(nely+1):
        for j in range(nelx+1):
            ax.text(x=j-0.5, y=i-0.3, 
                    s=",".join([str(s+ndoffset) for s in \
                                ndgrid[i*ndof:(i+1)*ndof,j]]), 
                    ha='center', va='center', fontsize=10)
    # put black dots at each node
    ax.scatter( np.repeat(np.arange(-0.5, nelx+0.5, 1),nely+1),
                np.tile(np.arange(-0.5, nely+0.5, 1),nelx+1),
                color="k",s=10, clip_on=False)
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
        plt.savefig("meshnumbering-"+["scalar","vector"][int(ndof!=1)]+"-2d.png",
                    format="png")
        plt.show()
    return

def plot_meshnumbering3d(nelx=4,nely=3,nelz=2,ndof=2):
    # create element and node numbering
    elgrid = create_nrs(nelx,nely,nelz)
    ndgrid = create_nrs(nelx+1,(nely+1)*ndof,nelz+1)
    #
    fig, axs = plt.subplots(ncols=2, nrows=nelz+1,figsize=(8, 6))
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
                         ha='center', va='center', fontsize=10)
    # add node numbers
    for i in range(nely+1):
        for j in range(nelx+1):
            for k in range(nelz+1):
                ax1.text(x=j-0.3, y=i-0.4,z=k-0.4,
                         s=",".join([str(s) for s in \
                                     ndgrid[k,i*ndof:(i+1)*ndof,j]]), 
                         ha='center', va='center', fontsize=10)
    # plt grid
    ax1.set_xticks(np.arange(0.5, nelx+0.5, 1), minor=False)
    ax1.set_yticks(np.arange(0.5, nely+0.5, 1), minor=False)
    ax1.set_zticks(np.arange(0.5, nelz+0.5, 1), minor=False)
    ax1.grid(which='major', color='black', linestyle='-', linewidth=1)
    # grid lines
    # x
    for j in np.arange(-0.5, nely+0.5, 1):
        for k in np.arange(-0.5, nelz+0.5, 1):
                ax1.plot(xs = np.array([-0.5, nelx-0.5]),
                         ys = np.array([j,j]),
                         zs = np.array([k,k]),
                         color="k")
    # y
    for i in np.arange(-0.5, nelx+0.5, 1):
        for k in np.arange(-0.5, nelz+0.5, 1):
            ax1.plot(xs = np.array([i, i]),
                     ys = np.array([-0.5, nely-0.5]),
                     zs = np.array([k,k]),
                     color="k")
    # z
    for i in np.arange(-0.5, nelx+0.5, 1):
        for j in np.arange(-0.5, nely+0.5, 1):
            ax1.plot(xs = np.array([i, i]),
                     ys = np.array([j, j]),
                     zs = np.array([-0.5, nelz-0.5]),
                     color="k")
    # remove ticks on x and y axis
    ax1.tick_params(axis='both',which='both',
                    bottom=False, labelbottom=False,
                    left=False, labelleft=False)
    # set axis limits
    ax1.set_xlim(-0.5, nelx-0.5)
    ax1.set_ylim(nely-0.5, -0.5)
    ax1.set_zlim(-0.5, nelz-0.5)
    # axis labels
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_zlabel("z", fontsize=12)
    # 2d plots 
    for i in range(nelz+1):
        axs[i,1].set_title("z = "+str(i), fontsize=12)
        plot_meshnumbering2d(nelx=nelx, nely=nely, ndof=ndof,
                             eloffset=i*nelx*nely,
                             ndoffset=i*(nelx+1)*(nely+1)*ndof,
                             ax=axs[i,1])
        axs[i,1].set_xlabel("x", fontsize=12)
        axs[i,1].set_ylabel("y", fontsize=12)
    # Adjust layout to prevent overlap
    fig.tight_layout()
    plt.savefig("meshnumbering-"+["scalar","vector"][int(ndof!=1)]+"-3d.png",
                format="png")
    plt.show()
    return

def show_local_ordering(ndim: int,
                        nodes: np.ndarray = np.array([[-1,-1,-1],
                                                      [ 1,-1,-1],
                                                      [ 1, 1,-1],
                                                      [-1, 1,-1],
                                                      [-1,-1, 1],
                                                      [ 1,-1, 1],
                                                      [ 1, 1, 1],
                                                      [-1, 1, 1]]),
                        box_nodes: np.ndarray = np.array([[-1,-1,-1],
                                                          [ 1,-1,-1],
                                                          [ 1, 1,-1],
                                                          [-1, 1,-1],
                                                          [-1,-1, 1],
                                                          [ 1,-1, 1],
                                                          [ 1, 1, 1],
                                                          [-1, 1, 1]]),
                        draw_axis: bool = True,
                        origin: np.ndarray = np.array([-1.3, -1.3, -1.3])
                        ) -> None:
    # reduce coordinates to the necessary ones given the dimension
    nodes = nodes[:int(2**ndim),:ndim]
    box_nodes = box_nodes[:int(2**ndim),:ndim]
    origin = origin[:ndim]
    #
    fig,ax = plt.subplots(1,1)
    if ndim == 3:
        gs = ax.get_gridspec()
        # remove the underlying Axes
        ax.remove()
        # 
        ax = fig.add_subplot(gs[:,0], projection='3d')
    # draw lines for box
    for i in range(ndim-1):
        nd = box_nodes[int(i*4):int((i+1)*4)]
        ax.plot(*np.split(np.append(nd,nd[0:1],axis=0),
                          ndim,axis=1),
                color="gray", zorder=1)
    if ndim==3:
        for i in range(4):
            ax.plot(*np.split(np.vstack((box_nodes[i:int(i+1),:],
                                         box_nodes[int(4+i):int(5+i),:])),
                              ndim,axis=1),
                    color="gray", zorder=1)
    # draw nodes
    ax.scatter(*np.split(nodes,ndim,axis=1),
               color="k", zorder=2)
    # label nodes
    for i, coord in enumerate(nodes):
        ax.text(*coord + np.array([0.1,0.1,0.1])[:ndim], 
                 f"{i}", fontsize=12, ha='right', va='bottom')
    #
    ax.tick_params(axis='both',
                   which='both',
                   bottom=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)
    ax.axis("off")
    # draw coordinate axis
    if draw_axis:
        if ndim == 2:
            arrow_dict = {"angles": "xy", 
                          "scale_units": "xy", 
                          "scale": 1}
        elif ndim == 3:
            arrow_dict = {"arrow_length_ratio": 0.1}
        
        ax.quiver(*origin, *[1, 0, 0][:ndim], 
                  **arrow_dict, color="blue")
        ax.text(*(np.array([1.1, 0, 0])[:ndim] + origin), 
                "x", fontsize=12,color="blue")
        
        ax.quiver(*origin, *[0, 1, 0][:ndim], 
                  **arrow_dict, color="red")
        ax.text(*(np.array([0, 1.1, 0])[:ndim] + origin), 
                "y", fontsize=12,color="red")
        
        if ndim == 3:
            ax.quiver(*origin, *[0, 0, 1],
                      **arrow_dict, color="green")
            ax.text(*(np.array([0, 0, 1.1])[:ndim] + origin), 
                    "z", fontsize=12,color="green")
    #
    plt.savefig(f"localnodeorder-scalar-{ndim}d.png",
                format="png")
    #
    plt.show()
    return
    
if __name__ == "__main__":
    # scalar fields
    #plot_meshnumbering2d(nelx=4,nely=3,ndof=1)
    #plot_meshnumbering3d(nelx=4,nely=3,ndof=1)
    # vector fields
    #plot_meshnumbering2d(nelx=3,nely=2,ndof=2)
    #plot_meshnumbering3d(nelx=3,nely=2,ndof=3)
    show_local_ordering(ndim = 2)
    show_local_ordering(ndim = 3)
