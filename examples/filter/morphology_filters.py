# SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np
import matplotlib.pyplot as plt

from topoptlab.filter.density_filter import DensityFilter
from topoptlab.filter.morphology_filter import DilateFilterSigmund2007,\
                                               ErodeFilterSigmund2007
from topoptlab.utils import map_eltoimg, map_imgtoel

def test_design(nelx, nely):
    """
    Black and white test design with features larger and smaller than the
    filter radius: a thick and a thin bar, a small solid island, a solid block
    with a small and a large hole.
    """
    img = np.zeros((nely, nelx))
    # thick bar
    img[4:12, 4:56] = 1.
    # thin bar
    img[18:20, 4:56] = 1.
    # small island
    img[28:30, 10:12] = 1.
    # block with small and large hole
    img[24:36, 24:56] = 1.
    img[29:31, 30:32] = 0.
    img[27:33, 42:50] = 0.
    return map_imgtoel(img[:, :, None], nelx=nelx, nely=nely)

if __name__ == "__main__":
    #
    nelx = 60
    nely = 40
    rmin = 2.5
    # large beta to come close to the exact max/min operators
    beta = 200.
    #
    kw = dict(nelx=nelx, nely=nely, n_constr=1, rmin=rmin)
    density = DensityFilter(**kw)
    dilate = DilateFilterSigmund2007(**kw)
    erode = ErodeFilterSigmund2007(**kw)
    D = lambda x: dilate.apply_filter(x, beta=beta)
    E = lambda x: erode.apply_filter(x, beta=beta)
    #
    x = test_design(nelx, nely)
    designs = {"original": x,
               "density": density.apply_filter(x),
               "dilate": D(x),
               "erode": E(x),
               "open (erode, dilate)": D(E(x)),
               "close (dilate, erode)": E(D(x)),
               "open-close": E(D(D(E(x)))),
               "close-open": D(E(E(D(x))))}
    #
    fig, axs = plt.subplots(4, 2, figsize=(5, 14))
    for ax, (name, xf) in zip(axs.flatten(), designs.items()):
        ax.imshow(map_eltoimg(xf, nelx=nelx, nely=nely)[:, :, 0],
                  cmap="gray_r", vmin=0., vmax=1.,
                  interpolation="none")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    plt.show()
