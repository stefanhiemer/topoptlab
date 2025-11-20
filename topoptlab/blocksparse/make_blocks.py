# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, List, Union

import numpy as np
from scipy.sparse import sparray

from topoptlab.output_designs import threshold

def create_equal_blocks(n_nodes : int, 
                        nblocks : int,
                        nnode_dof : int,
                        **kwargs : Any) -> List[np.ndarray]:
    """
    Create block indices by just splitting the dofs in nblocks even-sized 
    chunks.

    Parameters
    ----------
    n_nodes : int
        number of nodes.
    nblocks : int
        number of blocks.
    nnode_dof : int
        number of degrees of freedom per node.

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    return np.split(np.arange(n_nodes*nnode_dof, dtype=np.int32), 
                    np.arange(0, n_nodes,
                              np.rint(n_nodes/nblocks),
                              dtype=np.int32)[1:nblocks]*nnode_dof)

def create_volthresh_blocks(xPhys : np.ndarray,
                            volfrac : float, 
                            edofMat : np.ndarray,
                            **kwargs : Any) -> List[np.ndarray]:
    """
    Create block indices for two blocks by performing the volume preserving 
    thresholding which creates two blocks. All indices belonging to solid 
    elements represent one block while the rest represent the other block.

    Parameters
    ----------
    xPhys : np.ndarray
        element densities for topology optimization used for scaling the 
        material properties. shape (nel,1) 
    volfrac : float
        volume fraction.
    edofMat : np.ndarray 
        element degree of freedom matrix. shape (nel,n_nodedof*n_nodes)

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    # get thresholds densities and collect the nonzero elements 
    inds = threshold(xPhys=xPhys, volfrac=volfrac).nonzero()[0]
    # collect node indices of belonging to "full" elements 
    inds = np.unique(edofMat[inds].flatten())
    return [inds,np.setdiff1d( np.arange(edofMat.max()+1,dtype=np.int32), inds )]

def create_quantile_blocks(xPhys : np.ndarray,
                           quantiles : Union[List,np.ndarray], 
                           edofMat : np.ndarray,
                           **kwargs : Any) -> List[np.ndarray]:
    """
    Create block indices for two blocks by .

    Parameters
    ----------
    xPhys : np.array, shape (nel,1)
        element densities for topology optimization used for scaling the 
        material properties. 
    quantiles : list
        list of n-quantiles to build blocks. Must be between 0 and 1, so 0.5 
        is the median.
    edofMat : np.ndarray 
        element degree of freedom matrix. shape (nel,n_nodedof*n_nodes)

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    #
    if isinstance(quantiles, list):
        quantiles = np.array(quantiles)
    elif isinstance(quantiles, np.ndarray):
        pass 
    else:
        raise ValueError("quantiles must be either list or np.ndarray.")
    # get sorted element indices
    el_inds = np.flip(np.argsort(xPhys[:,0]))
    # get divisions by quantiles of sorted element indices
    qt = np.rint(quantiles*xPhys.shape[0]).astype(int)
    qt = np.hstack( (np.array([0]),qt,np.array([el_inds.shape[0]])),
                   dtype=int)
    # construct each block, but there might be overlap between blocks
    blocks = [np.unique(edofMat[el_inds[q1:q2],:].flatten(),
                        return_counts=True) \
              for q1,q2 in zip(qt[:-1],qt[1:])]
    #
    block_nd_inds, block_nd_counts = zip(*blocks)
    block_nd_inds = list(block_nd_inds)
    nd_inds = np.arange(edofMat.max(),dtype=int)
    masks = [np.isin(nd_inds,block_nd_ind) for block_nd_ind in block_nd_inds]
    #
    done = np.zeros(nd_inds.shape,dtype=bool)
    i = 0
    for i,mask in enumerate(masks):
        block_nd_inds[i] = np.setdiff1d(block_nd_inds[i], nd_inds[done])
        done[mask] = True
        i = i + 1
    return block_nd_inds