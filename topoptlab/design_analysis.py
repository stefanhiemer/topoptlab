import numpy as np

from topoptlab.utils import map_eltoimg, map_eltovoxel
from topoptlab.utils import map_imgtoel, map_voxeltoel


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