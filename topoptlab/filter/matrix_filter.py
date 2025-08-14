from typing import Union,Any,Tuple

import numpy as np
from scipy.sparse import coo_matrix,csc_matrix


def assemble_matrix_filter(nelx: int, nely: int, rmin: float,
                           nelz: Union[int, None] = None,
                           ndim: int = 2,
                           **kwargs: Any) -> Tuple[csc_matrix,np.matrix]:
    """
    Assemble distance based filters as sparse matrix that is applied on to
    densities/sensitivities by standard multiplication.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    rmin : float
        cutoff radius for the filter. Only elements within the element-center
        to element center distance are used for filtering.
    nelz : int or None
        number of elements in z direction. Ignored if ndim < 3.
    ndim : int
        number of dimensions

    Returns
    -------
    H : csc matrix
        unnormalized filter.
    Hs : np matrix
        normalization factor.

    """
    # number of elements/densities
    if ndim == 2:
        n = nelx*nely
    elif ndim == 3:
        n = nelx*nely*nelz
    # index array of densities/elements
    el = np.arange(n)
    # filter size
    nfilter = int(n*((2*(np.ceil(rmin)-1)+1)**ndim))
    # create empty arrays for indices and values of final filter
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    # find coordinates of each element
    if ndim == 2:
        x,y = np.divmod(el,nely) # same as np.floor(el/nely),el%nely
    elif ndim == 3:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
    # find coordinates of neighbours for each element
    kk1 = np.maximum(x-(np.ceil(rmin)-1), 0).astype(int)
    kk2 = np.minimum(x+np.ceil(rmin), nelx).astype(int)
    ll1 = np.maximum(y-(np.ceil(rmin)-1), 0).astype(int)
    ll2 = np.minimum(y+np.ceil(rmin), nely).astype(int)
    if ndim == 3:
        mm1 = np.maximum(z-(np.ceil(rmin)-1), 0).astype(int)
        mm2 = np.minimum(z+np.ceil(rmin), nelz).astype(int)
    n_neigh = (kk2-kk1)*(ll2-ll1)
    if ndim ==3:
        n_neigh = n_neigh * (mm2-mm1)
    el = np.repeat(el, n_neigh)
    x = np.repeat(x, n_neigh)
    y = np.repeat(y, n_neigh)
    if ndim == 3:
        z = np.repeat(z, n_neigh)
    cc = np.arange(el.shape[0])
    if ndim == 2:
        # coordinate of neighbor
        xn,yn = np.hstack([np.stack([a.flatten() for a in \
                         np.meshgrid(np.arange(k1,k2),np.arange(l1,l2))]) \
                         for k1,k2,l1,l2 in zip(kk1,kk2,ll1,ll2)])
        # hat function
        fac = rmin-np.sqrt((x-xn)**2+(y-yn)**2)
    elif ndim == 3:
        # coordinate of neighbor
        xn,yn,zn = np.hstack([np.stack([a.flatten() for a in \
                              np.meshgrid(np.arange(k1,k2),
                                          np.arange(l1,l2),
                                          np.arange(m1,m2))]) \
                           for k1,k2,l1,l2,m1,m2 in \
                           zip(kk1,kk2,ll1,ll2,mm1,mm2)])
        # hat function
        fac = rmin-np.sqrt((x-xn)**2+(y-yn)**2+(z-zn)**2)
    iH[cc] = el # row
    if ndim == 2:
        jH[cc] = nely*xn+yn #column
    elif ndim == 3:
        jH[cc] = (zn*nelx + xn)*nely + yn
    sH[cc] = np.maximum(0.0, fac)
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(n, n)).tocsc()
    # normalization constants
    Hs = H.sum(1)
    return H,Hs