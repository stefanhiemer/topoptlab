from typing import Callable, Tuple, Union, Any

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from topoptlab.fem import create_matrixinds
from topoptlab.elements.screenedpoisson_2d import lk_screened_poisson_2d
from topoptlab.elements.screenedpoisson_3d import lk_screened_poisson_3d
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d


def assemble_helmholtz_filter(nelx: int, nely: int, rmin: float,
                              nelz: Union[int, None] = None, 
                              l: np.ndarray = np.array([1.,1.]),
                              n1: Union[None, np.ndarray] = None,
                              n2: Union[None, np.ndarray] = None,
                              n3: Union[None, np.ndarray] = None, 
                              n4: Union[None, np.ndarray] = None,
                              **kwargs: Any)  -> Tuple[csc_matrix,csc_matrix]:
    """
    Assemble Helmholtz PDE based filter from "Efficient topology optimization
    in MATLAB using 88 lines of code".

    This filter works by mappinging the element densities to nodes via the
    operator TF, then performing the actual filter operation by solving a
    Helmholtz / screened Poisson PDE in the standard FEM style with subsequent
    back-mapping of the nodal (filtered) densities to the elements.

    Parameters
    ----------
    nelx : int
        number of elements in x direction.
    nely : int
        number of elements in y direction.
    volfrac : float
        volume fraction.
    rmin : float
        cutoff radius for the filter.
    nelz : int or None
        number of elements in z direction.
    n1 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    n2 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    n3 : np.ndarray or None
        index array to help constructing the stiffness matrix.
    n4 : np.ndarray or None
        index array to help constructing the stiffness matrix.

    Returns
    -------
    KF : csc matrix
        stiffness matrix.
    TF : csc matrix
        mapping (or in this special case averaging) operator that maps element
        densities to nodes and its inverse maps nodal densities back to the
        elements.

    """
    if nelz is None:
        ndim = 2
        element = lk_screened_poisson_2d
        create_edofMat = create_edofMat2d
    else:
        ndim = 3
        raise NotImplementedError("3D not yet completely implemented.")
        element = lk_screened_poisson_3d
        create_edofMat = create_edofMat3d
    # element indices
    nel = np.prod([nelx,nely,nelz][:ndim])
    el = np.arange(nel)
    #
    if n1 is None:
        elx,ely = np.arange(nelx)[:,None], np.arange(nely)[None,:]
        n1 = ((nely+1)*elx+ely).flatten()
        n2 = ((nely+1)*(elx+1)+ely).flatten()
    # conversion of filter radius via Green's function
    Rmin = rmin/(2*np.sqrt(3))
    # collect element stiffness matrix
    KE = element(k=Rmin**2, l=l)
    # total number of degrees of freedom
    ndof = np.prod(np.array([nelx,nely,nelz])[:ndim]+1)
    # element degree of freedom matrix
    edofMat, n1, n2, n3, n4 = create_edofMat(nelx=nelx,nely=nely,nelz=nelz,
                                               nnode_dof=1)
    iM,jM = create_matrixinds(edofMat,mode="full")
    sM = np.tile(KE.flatten(),nelx*nely)
    KF = coo_matrix((sM, (iM, jM)), shape=(ndof, ndof)).tocsc()
    # operator that maps element densities to nodes
    iTF = edofMat.flatten(order='F')
    jTF = np.tile(el, 4)
    sTF = np.full(4*nelx*nely,1/4)
    TF = coo_matrix((sTF, (iTF, jTF)), shape=(ndof,nelx*nely)).tocsc()
    return KF,TF
