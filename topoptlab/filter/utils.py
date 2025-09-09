# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Union,Callable,Dict

import numpy as np

from topoptlab.geometries import diracdelta
from topoptlab.utils import map_eltoimg,map_eltovoxel

from matplotlib.pyplot import subplots,figure,figaspect,show

def visualise_filter(n: int,
                     apply_filter: Callable,
                     geo: Union[None,np.ndarray,Callable] = None,
                     fig_kws: Union[None,Dict] = None)->None:
    """
    Apply filter to a given geometry and display the original geometry next to
    the filtered one in order to understand the effect of a filter on a given
    geometry of design densities.

    Parameters
    ----------
    n : tuple
        contains number of elements in x,y and z direction depending on number
        of dimensions.
    apply_filter : callable
        function that applies filter.
    geo : callable or np.ndarray of shape(np.prod(n)) or None
        geometry of design densities on which to apply filter.
    fig_kws : dict or None, optional
        keywords for figure.

    Returns
    -------
    None.

    """
    #
    ndim = len(n)
    #
    nelx,nely,nelz = n[:ndim] + (None,None,None)[ndim:]
    #
    if geo is None:
        geo = diracdelta(nelx=nelx ,nely=nely, nelz=nelz,
                         location=None )[:,None]
    elif callable(geo):
        geo = diracdelta(nelx=nelx ,nely=nely, nelz=nelz,
                         location=None )
    elif isinstance(geo, np.ndarray):
        geo.shape[0] = int(np.prod(n))
    #
    if ndim == 2:
        # default plot settings 2d
        if fig_kws is None:
            fig_kws = {"figsize": (8,8)}
        #
        fig,axs = subplots(1,2,**fig_kws)
        #
        axs[0].imshow(1-map_eltoimg(quant=geo,
                                    nelx=nelx, nely=nely),
                      cmap="grey")
        #
        filtered = map_eltoimg(quant=apply_filter(geo),
                                    nelx=nelx, nely=nely)
        axs[1].imshow(1-filtered,
                      cmap="grey")
        for i in range(2):
            axs[i].tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelbottom=False,
                               labelleft=False)
            axs[i].axis("off")
    elif ndim == 3:
        # default plot settings 3d
        if fig_kws is None:
            fig_kws = {"figsize": figaspect(2.)}
        #
        fig = figure(**fig_kws)
        # unfiltered
        axs = []
        axs.append(fig.add_subplot(2, 1, 1, projection='3d'))
        axs.append(fig.add_subplot(2, 1, 2, projection='3d'))
        #
        dirac_voxel = map_eltovoxel(geo,
                                    nelx=nelx, nely=nely, nelz=nelz)
        #
        facecolors = np.ones(dirac_voxel.shape[:-1] + (4,))
        facecolors[:,:,:,:-1] = 1 - dirac_voxel
        facecolors[:,:,:,-1] = dirac_voxel[:,:,:,0]
        #
        axs[0].voxels(filled = ~np.isclose(dirac_voxel[:,:,:,0], 0),
                      facecolors=facecolors)
        # filtered
        filtered_voxel = map_eltovoxel(quant=apply_filter(geo),
                                       nelx=nelx, nely=nely, nelz=nelz)
        #
        facecolors = np.ones(filtered_voxel.shape[:-1] + (4,))
        facecolors[:,:,:,:-1] = 1 - filtered_voxel
        facecolors[:,:,:,-1] = filtered_voxel[:,:,:,0]
        #
        axs[1].voxels(filled = ~np.isclose(filtered_voxel[:,:,:,0], 0),
                      facecolors=facecolors)
        #
        for i in range(2):
            # limits
            for j,nel in enumerate(n):
                axs[i].set_xlim(0,nel)
            #
            axs[i].set_xlabel( "z" )
            axs[i].set_ylabel( "y" )
            axs[i].set_zlabel( "x" )
        #
    print("mass before filter operation: ", geo.sum(),"\n",
          "mass after filter operation: ", filtered_voxel.sum())

    show()
    return
