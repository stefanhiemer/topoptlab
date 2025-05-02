import numpy as np
from scipy.ndimage import grey_opening,grey_closing,zoom

from topoptlab.geometries import sphere, ball
from topoptlab.utils import map_eltoimg, map_eltovoxel
from topoptlab.utils import map_imgtoel, map_voxeltoel

def upsampling(x, magnification,
               nelx,nely,nelz=None,
               return_flat=True, order=0):
    """
    Upsample current design variables defined on the standard regular grid to 
    a larger design by interpolation. With order 0 the design is replicated on
    a finer scale in a volume conserving fashion, otherwise spline 
    interpolation might violate this.

    Parameters
    ----------
    x : np.ndarray shape (n)
        design variables.
    magnification : float
        magnification factor.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None, optional
        number of elements in z direction. The default is None.
    return_flat : bool, optional
        return the design variables flattened. If false returns an image or a 
        voxel graphic. The default is True.
    order : int, optional
        order of spline interpolation for upsampling. The default is 0.

    Returns
    -------
    x_new : np.ndarray shape (n) or shape (nely,nelx) or shape (nelz,nely,nelx)
        upsampled design variables.

    """
    
    if nelz is None:
        x = map_eltoimg(quant=x, nelx=nelx, nely=nely)
    else:
        x = map_eltovoxel(quant=x, nelx=nelx, nely=nely, nelz=nelz)
    #
    x = zoom(x,zoom=magnification,
             order=order,mode="nearest",cval=0.)
    #
    if return_flat:
        if nelz is None:
            x = map_imgtoel(quant=x, nelx=nelx, nely=nely)
        else:
            x = map_voxeltoel(quant=x, nelx=nelx, nely=nely, nelz=nelz)
    return x

def lengthscale_violations(x,r,nelx,nely,nelz=None):
    """
    Visualize length scale violations as suggested in chapter 2.6 of 
    
    Sigmund, Ole. "On benchmarking and good scientific practise in topology 
    optimization." Structural and Multidisciplinary Optimization 65.11 (2022): 
    315.
    
    Length scale violations are determined by creating a sphere of radius r 
    and performing morphological closing and opening operations. For length 
    scale violations in the void we perform 
    
    close(x) - x
    
    while for length scale violation in solid material we do
    
    x - open(x)

    Parameters
    ----------
    x : np.ndarray shape (n)
        design variables.
    r : float
        radius of sphere that determines length scale.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None, optional
        number of elements in z direction. The default is None.

    Returns
    -------
    x_new : np.ndarray shape (n) or shape (nely,nelx) or shape (nelz,nely,nelx)
        upsampled design variables.

    """
    
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
        structure = ball(nelx=l, nely=l, nelz=l,
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
