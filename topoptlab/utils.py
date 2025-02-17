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

def unique_sort(iM,jM,combine=False):
    """
    Sort first according to iM, then sort values of equal value iM according 
    to jM.

    Parameters
    ----------
    iM : np.ndarray, shape (n)
        first array.
    jM : np.ndarray, shape (n)
        second array.
    combine : scalar bool
        if True, stack both to to a column array of shape (n,2) 

    Returns
    -------
    iM : np.ndarray shape (n)
        if not combine returns sorted iM
    jM : np.ndarray shape (n)
        if not combine returns sorted jM
    M : np.ndarray shape (n,2)
        if combine returns column stack of sort iM and jM.

    """
    
    inds = np.lexsort((jM,iM))
    if combine:
        return np.column_stack((iM[inds],jM[inds])) 
    else:
        return iM[inds],jM[inds]

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

def map_imgtoel(img,nelx,nely,**kwargs):
    """
    Map image of quantity back to 1D np.ndarray with correct (!) ordering.

    Parameters
    ----------
    img : np.ndarray, shape (nely,nelx)
        image of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.

    Returns
    -------
    quant : np.ndarray, shape (n)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely])+img.shape[2:]
    return img.reshape(shape,order="F")

def map_eltovoxel(quant,nelx,nely,nelz,**kwargs):
    """
    Map quantity located on elements on the usual regular grid to a voxels.

    Parameters
    ----------
    quant : np.ndarray, shape (n,nchannel)
        some quantity defined on each element (e. g. element density).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    voxel : np.ndarray shape (nelz,nely,nelx,nchannel)
        quantity mapped to voxels.

    """
    #
    shape = (nelz,nelx,nely)+quant.shape[1:]
    return quant.reshape(shape).transpose((0,2,1)+tuple(range(3,len(shape))))

def map_voxeltoel(voxel,nelx,nely,nelz,**kwargs):
    """
    Map voxels of quantity back to on elements on the usual regular grid.

    Parameters
    ----------
    voxel : np.ndarray, shape (nelz,nely,nelx,nchannel)
        voxels of quantity (e. g. of element densities).
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    quant : np.ndarray, shape (n,nchannel)
        quantity mapped back to 1D np.ndarray with hopefully correct ordering.

    """
    shape = tuple([nelx*nely*nelz])+voxel.shape[3:]
    voxel = voxel.transpose((0,2,1)+tuple(range(3,len(voxel.shape))))
    return voxel.reshape(shape)

def elid_to_coords(el,nelx,nely,nelz=None,**kwargs):
    """
    Map element ids to cartesian coordinates in the usual regular grid.

    Parameters
    ----------
    el : np.ndarray, shape (n)
        elment IDs.
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    nelz : int or None
        number of elements in z direction.

    Returns
    -------
    img : np.ndarray shape (nely,nelx)
        quantity mapped to image.

    """
    # find coordinates of each element/density
    if nelz is None:
        x,y = np.divmod(el,nely) # same as np.floor(el/nely),el%nely
        return x,y
    else:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
        return x,y,z