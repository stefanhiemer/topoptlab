# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,List,Tuple,Union

import numpy as np
from scipy.sparse import coo_matrix,csc_matrix

from topoptlab.filter.filter import TOFilter
from topoptlab.filter.kernels import hat_kernel

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
                 n_constr: int,
                 rmin: float,
                 kernel_fn: Callable,
                 nelz: Union[int, None] = None,
                 pbc: Union[bool, List, np.ndarray] = False,
                 filter_objective: bool = True,
                 constraint_filter_mask: Union[None, np.ndarray] = None,
                 el_flags: Union[None, np.ndarray] = None,
                 el_flags_policy: Union[None, dict] = None,
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
        n_constr : int
            number of constraints.
        rmin : float
            cutoff radius for the filter.
        kernel_fn : callable
            weighting kernel passed to the underlying filter assembly.
        nelz : int or None
            number of elements in z direction.
        pbc : bool, list or np.ndarray
            flag for periodic boundary conditions.
        filter_objective : bool
            if True, filter is applied to objective sensitivities.
        constraint_filter_mask : None or np.ndarray of shape (n_constr,)
            if None, filter is applied to all constraint sensitivities.
            Otherwise, a boolean array indicating which constraint
            sensitivities are filtered.
        el_flags : None or np.ndarray
            array of element flags (0 free, 1 passive, 2 active).
        el_flags_policy : None or dict
            policy dict controlling filter behaviour for prescribed elements.

        Returns
        -------
        None

        """
        self.H, self.Hs = assemble_matrix_filter(nelx=nelx,
                                                  nely=nely,
                                                  kernel_fn=kernel_fn,
                                                  nelz=nelz,
                                                  rmin=rmin,
                                                  pbc=pbc,
                                                  el_flags=el_flags,
                                                  el_flags_policy=el_flags_policy)
        self._filter_objective = filter_objective
        if constraint_filter_mask is None:
            self._constraint_filter_mask = np.ones(n_constr, dtype=bool)
        elif isinstance(constraint_filter_mask, np.ndarray) and \
                constraint_filter_mask.shape == (n_constr,):
            self._constraint_filter_mask = constraint_filter_mask
        else:
            raise TypeError("constraint_filter_mask must be None or np.ndarray of shape (n_constr,): ", constraint_filter_mask)
        return
        
    def apply_filter(self, 
                     x: np.ndarray, 
                     **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the (intermediate) design variables x:
            
            x_filtered = np.asarray(H*(dobj/Hs))
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables, shape (n, k) where k >= 1.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables, shape (n, k).

        """
        return np.asarray(self.H*x/self.Hs)
        
    def apply_filter_dx(self, 
                        dx_filtered: np.ndarray, 
                        **kwargs: Any) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule assuming
        
        x_filtered = filter(x)
        
        to get the sensitivities with respect to the (unfiltered) design 
        variables or in the case of many filters intermediate design variables:
            
            dx = H@dx_filtered / Hs
        
        Parameters
        ----------
        dx_filtered : np.ndarray
            sensitivities with respect to filtered design variables,
            shape (n, k).

        Returns
        -------
        dx : np.ndarray
            design sensitivities with respect to un-filtered design variables,
            shape (n, k).
        """
        return np.asarray(self.H*(dx_filtered/self.Hs))
    
    @property
    def vol_conserv(self) -> bool:
        """
        Set self.vol_conserv to False as filter is not necessarily volume
        conserving.

        Returns
        -------
        False
        """
        return False

    @property
    def filter_objective(self) -> bool:
        """
        If True, filter is applied to objective sensitivities.
        """
        return self._filter_objective

    @property
    def constraint_filter_mask(self) -> bool:
        """
        Default scalar mask for constraint sensitivity filtering.
        """
        return self._constraint_filter_mask

    @property
    def changes_filter_kw(self) -> bool:
        return False

    def update_filter_kw(self, filter_kw: dict) -> None:
        return

def assemble_matrix_filter(nelx: int,
                           nely: int,
                           rmin: Union[float,List,np.ndarray],
                           nelz: Union[int, None] = None,
                           pbc: Union[bool,List,np.ndarray] = False,
                           el_flags: Union[None, np.ndarray] = None,
                           el_flags_policy: Union[None, dict] = None,
                           kernel_fn: Union[Callable, None] = None,
                           compute_coords: bool = True,
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
    rmin : float, list or np.ndarray
        cutoff radius for the filter. Only elements within the element-center
        to element center distance are used for filtering. Can also be defined
        for each dimension.
    nelz : int or None
        number of elements in z direction.
    pbc : bool, list or np.ndarray
        flag for periodic boundary conditions.
    kernel_fn : callable
        function that maps stencil offset coordinates to weights. When
        compute_coords is True (default), called as kernel_fn(x, y, rmin) in
        2D or kernel_fn(x, y, rmin, z) in 3D with the stencil offset arrays
        and the effective scalar rmin. When compute_coords is False, called as
        kernel_fn(rmin) and must return the weight array for all stencil
        offsets directly.
    compute_coords : bool
        if True (default), stencil offset coordinate arrays are computed and
        passed to kernel_fn. If False, kernel_fn is called with rmin only.

    Returns
    -------
    H : csc.matrix
        unnormalized filter.
    Hs : np.matrix
        normalization factor.

    """
    # number of elements/design variables
    if nelz is None:
        ndim = 2
    else:
        ndim = 3
    #
    n = np.array([nelx,nely,nelz][:ndim])
    nel= np.prod(n)
    #
    if isinstance(pbc,bool):
        pbc = np.array(ndim*[pbc])
    elif isinstance(pbc, list):
        pbc = np.array(pbc)
    elif isinstance(pbc, np.ndarray):
        if len(pbc.shape)==1 and pbc.shape[0] == 1:
            pbc = np.array(ndim*[pbc[0]])
        elif pbc.shape != ndim:
            raise ValueError("pbc must be of shape (1) or (ndim): ",
                             pbc.shape)
    else:
        raise ValueError("pbc must be list,np.ndarray or bool: ",
                         pbc)    
    #
    if isinstance(rmin, float):
        rmin = np.array(ndim*[rmin])
    elif isinstance(rmin, list):
        rmin = np.array(rmin)
    elif isinstance(rmin, np.ndarray):
        if len(rmin.shape)==1 and rmin.shape[0] == 1:
            rmin = np.array(ndim*[rmin[0]])
        elif rmin.shape != ndim:
            raise ValueError("rmin must be of shape (1) or (ndim): ",
                             rmin.shape)
    else:
        raise ValueError("rmin must be list,np.ndarray or float: ",
                         rmin)
    # index array of densities/elements
    el = np.arange(nel)
    # filter size
    nfilter = int(nel*( np.prod(2*(np.ceil(rmin)-1)+1) ))
    # create empty arrays for indices and values of final filter
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    # find coordinates of each element
    if ndim == 2:
        coords = np.column_stack(np.divmod(el,nely)) # same as np.floor(el/nely),el%nely
    elif ndim == 3:
        z,rest = np.divmod(el,nelx*nely)
        x,y = np.divmod(rest,nely)
        coords = np.column_stack( (np.divmod(rest,nely)+tuple([z])) )
    # convert/round rmin to integer 
    dx = np.maximum(np.ceil(rmin)-1,np.zeros(ndim)).astype(int)
    # create stencil for neighbors
    neighbors = [np.arange(-d,d+1) for d in dx]
    neighbors = [n.flatten() for n in np.meshgrid(*neighbors, indexing='ij')]
    neighbors = np.column_stack(neighbors)
    # compute kernel weights for each stencil offset
    effective_rmin = np.linalg.norm(rmin, ord=2) / np.sqrt(ndim)
    if compute_coords:
        if ndim == 2:
            sH_stencil = kernel_fn(x=neighbors[:, 0], y=neighbors[:, 1],
                                   z=None,
                                   rmin=effective_rmin)
        else:
            sH_stencil = kernel_fn(x=neighbors[:, 0], y=neighbors[:, 1],
                                   rmin=effective_rmin, z=neighbors[:, 2])
    else:
        sH_stencil = kernel_fn(rmin=effective_rmin)
    # create coordinates of neighbours for each element
    neighbors = coords[:,None,:] + neighbors[None,:,:]
    # apply boundary condtions
    # finite
    if not np.any(pbc):
        inside = np.all((neighbors >= 0) &\
                        (neighbors < n[None,None,:]),
                        axis=2)
    # pbc
    else:
        #
        length = ( np.array([nelx,nely,nelz][:ndim]) )[None,None,:]
        #
        out_of_domain_neg = neighbors < 0 
        out_of_domain_pos = neighbors >= length
        #
        inside = np.all(~(out_of_domain_neg | out_of_domain_pos) |\
                        pbc[None,None,:],
                        axis=2)
        #
        length = np.ones(out_of_domain_neg.shape)*length
        #
        mask = out_of_domain_neg&pbc[None,None,:]
        neighbors[mask]=neighbors[mask]+length[mask]
        #
        mask = out_of_domain_pos&pbc[None,None,:]
        neighbors[mask]=neighbors[mask]-length[mask]
    # convert neighbor coordinates to indices
    neighbors = (neighbors*\
                 np.array([nely,1,nelx*nely][:ndim])[None,None,:]).sum(axis=-1)
    #
    iH = np.repeat(el, np.sum(inside,axis=1))
    jH = neighbors[inside].astype(int)
    sH = np.tile(sH_stencil, (nel,1))[inside]
    # eliminate passive (1) and active (2) elements from filter matrix;
    # non-design elements (flag=3) are intentionally kept in the stencil.
    if el_flags is not None and el_flags_policy is not None and \
            el_flags_policy["neglect_in_filter"]:
        prescribed = (el_flags == 1) | (el_flags == 2)
        keep = ~(prescribed[iH.astype(int)] | prescribed[jH.astype(int)])
        iH, jH, sH = iH[keep], jH[keep], sH[keep]
    # Finalize assembly and convert to csc format
    H = coo_matrix((sH, (iH, jH)), shape=(nel, nel)).tocsc()
    # normalization constants
    Hs = H.sum(1)
    # set to 1 for active/passive elements to avoid zero division.
    if el_flags is not None and el_flags_policy is not None and \
            el_flags_policy["neglect_in_filter"]:
        Hs[(el_flags == 1) | (el_flags == 2)] = 1.
    return H,Hs

def assemble_matrix_filter_legacy(nelx: int, nely: int, rmin: float,
                                  nelz: Union[int, None] = None,
                                  **kwargs: Any
                                  ) -> Tuple[csc_matrix,np.matrix]:
    """
    Will only be used for testing. Assemble distance based filters as sparse 
    matrix that is applied on to densities/sensitivities by standard 
    multiplication. 

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
    H : csc.matrix
        unnormalized filter.
    Hs : np.matrix
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
    return H, Hs