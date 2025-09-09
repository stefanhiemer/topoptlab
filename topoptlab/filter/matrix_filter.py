# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Tuple,Union

import numpy as np
from scipy.sparse import coo_matrix,csc_matrix

from topoptlab.filter.filter import TOFilter

class MatrixFilter(TOFilter):
    """
    Implementation here is based on the implementation in 
    
    Andreassen, Erik, et al. "Efficient topology optimization in MATLAB using 
    88 lines of code." Structural and Multidisciplinary Optimization 43.1 
    (2011): 1-16.
    
    but extended to 3D.
    """
    
    def __init__(self,
                 nelx: int, 
                 nely: int, 
                 rmin: float,
                 nelz: Union[int, None] = None, 
                 **kwargs: Any) -> None:
        """
        Assemble matrix-based filter from "Efficient topology optimization in 
        MATLAB using 88 lines of code".
        
        Parameters
        ----------
        nelx : int
            number of elements in x direction.
        nely : int
            number of elements in y direction.
        rmin : float
            cutoff radius for the filter.
        nelz : int or None
            number of elements in z direction.
        
        Returns
        -------
        None

        """
        self.H, self.Hs = assemble_matrix_filter(nelx=nelx, 
                                                 nely=nely, 
                                                 nelz=nelz,
                                                 rmin=rmin)
        
    def apply_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply filter to the (intermediate) design variables x:
            
            x_filtered = np.asarray(H*(dobj/Hs))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return np.asarray(self.H*(x/self.Hs))
        
    def apply_filter_dx(self, dx_filtered: np.ndarray) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = H@dx_filtered / Hs
        
        Parameters
        ----------
        x_filtered : np.ndarray
            filtered design variables.
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables.
            
        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables.
        """
        return np.asarray(self.H*(dx_filtered/self.Hs))

def assemble_matrix_filter(nelx: int, nely: int, rmin: float,
                           nelz: Union[int, None] = None,
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
        number of elements in z direction.

    Returns
    -------
    H : csc matrix
        unnormalized filter.
    Hs : np matrix
        normalization factor.

    """
    # number of elements/design variables
    if nelz is None:
        ndim = 2
        n = nelx*nely
    else:
        ndim = 3
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