import numpy as np

from topoptlab.geometries import sphere, ball
from topoptlab.utils import map_eltoimg, map_eltovoxel
from topoptlab.utils import map_imgtoel, map_voxeltoel

def lengthscale_violations(x,nelx,nely,r,nelz=None):
    #
    r = int(r)
    l = 1+int(2*r)
    #
    if nelz is None:
        x = map_eltoimg(quant=x, nelx=nelx, nely=nely)
        structure = sphere(nelx=l, nely=l,
                           center=(np.median([0,l-1]),
                                   np.median([0,l-1])),
                           radius=r, fill_value=1.)
        structure = map_eltoimg(quant=structure, nelx=l, nely=l)
    else:
        x = map_eltovoxel(quant=x, nelx=nelx, nely=nely, nelz=nelz)
        structure = ball(nelx=l, nely=l, nelz,=l,
                           center=(np.median([0,l-1]),
                                   np.median([0,l-1]),
                                   np.median([0,l-1])),
                           radius=r, fill_value=1.)
        structure = map_eltovoxel(quant=structure, nelx=l, nely=l, nelz=l)
    #
    solidviolation = x-grey_opening(x,size=structure.shape,
                                    structure=structure,
                                    mode="nearest",cval=0.)
    voidviolation = grey_closing(x,size=structure.shape,
                                 structure=structure,
                                 mode="nearest",cval=0.)-x
    return solidviolation, voidviolation

def map_eltoimg(quant,nelx,nely,**kwargs):
    """
    Map quantity located on elements on the usual regular grid to an image.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.

    Returns
    -------
    img : np.ndarray, shape (nely,nelx,nchannel)
        quantity mapped to image.

    """
    #
    shape = (nely,nelx)+quant.shape[1:]
    return quant.reshape(shape,order="F")
