import numpy as np

def rotation_matrix(theta):
    """
    2D rotation matrix

    Parameters
    ----------
    theta : np.ndarray, shape (n)
        angle in radian.

    Returns
    -------
    R : np.ndarray shape (n,2,2)
        rotation matrices.

    """
    return np.column_stack((np.cos(theta),-np.sin(theta),
                            np.sin(theta),np.cos(theta)))\
          .reshape((theta.shape[0],2,2))
          
def map_toimg(quant,nelx,nely,**kwargs):
    """
    Map densities located on the standard regular grid to an image.

    Parameters
    ----------
    quant : np.ndarray, shape (n)
        some quantity defined on each element (e. g. element density).

    Returns
    -------
    img : np.ndarray shape (nely,nelx)
        quantity mapped to image.

    """
    
    return quant.reshape((nelx, nely)).T 

def map_tovoxel(quant,nelx,nely,nelz,**kwargs):
    """
    Map densities located on the standard regular grid to a voxel graphic.

    Parameters
    ----------
    quant : np.ndarray, shape (n)
        some quantity defined on each element (e. g. element density).

    Returns
    -------
    img : np.ndarray shape (nely,nelx)
        quantity mapped to image.

    """
    
    return quant.reshape(nelz,nelx, nely).transpose(0,2,1)

def elid_to_coords(el,nelx,nely,nelz=None,**kwargs):
    """
    Map densities located on the standard regular grid to a voxel graphic.

    Parameters
    ----------
    quant : np.ndarray, shape (n)
        some quantity defined on each element (e. g. element density).

    Returns
    -------
    img : np.ndarray shape (nely,nelx)
        quantity mapped to image.

    """
    # find coordinates of each element/density
    if nelz is None:
        x,y = np.divmod(el,nely) # same as np.floor(el/nely),el%nely
    else:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
    return 