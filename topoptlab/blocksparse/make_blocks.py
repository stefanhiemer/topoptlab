# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, List

import numpy as np
from scipy.sparse import sparray

from topoptlab.output_designs import threshold

def create_equal_blocks(A: sparray, 
                        nblocks: int,
                        **kwargs) -> List[np.ndarray]:
    """
    Create block indices by just splitting the dofs in nblocks even-sized 
    chunks.

    Parameters
    ----------
    A : sparray
        matrix to be solved.
    nblocks : int
        function to solve/factorize each block.

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    return np.split(np.arange(A.shape[0]), 
                    np.arange(0, A.shape[0],
                              np.ceil(A.shape[0]/nblocks).astype(int))[1:])

def create_volthresh_blocks(xPhys : np.ndarray,
                            volfrac : float, 
                            edofMat : np.ndarray,
                            **kwargs) -> List[np.ndarray]:
    """
    Create block indices for two blocks by performing the volume preserving 
    thresholding.

    Parameters
    ----------
    xPhys : np.array, shape (nel)
        element densities for topology optimization used for scaling the 
        material properties. 
    volfrac : float
        volume fraction.

    Returns
    -------
    block_inds : List[np.ndarray]
        list containing of indices for each block.

    """
    inds = threshold(xPhys=xPhys, volfrac=volfrac).nonzero()[0]
    inds = np.unique(edofMat[inds].flatten())
    return [inds,np.setdiff1d( np.arange(edofMat.max()+1,dtype=np.int32), inds )]