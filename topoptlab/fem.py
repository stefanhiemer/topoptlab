# SPDX-License-Identifier: GPL-3.0-or-later
from abc import ABC, abstractmethod
from typing import Callable,List,Tuple,Union

from itertools import product

import numpy as np
from scipy.sparse import coo_array,csc_array,sparray

from cvxopt import spmatrix

from topoptlab.elements.bilinear_quadrilateral import shape_functions

class FEM_Phys(ABC):
    """
    Base class for different FEM physical problems.
    """
    
    @abstractmethod
    def __init__(self) -> None:
        """
        Initialize the FEM problem.
        
        Returns
        -------
        None

        """
        ...
        
    @abstractmethod 
    def assemble_system(self):
        """
        Assemble linear system for the global matrix.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod
    def bc(self):
        """
        Apply boundary conditions to linear system.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod
    def coupling(self):
        """
        Apply boundary conditions to linear system.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod
    def to_interpolation(self):
        """
        Problem specific interpolation.
        
        Returns
        -------
        None
        
        """
        ...
        
    @abstractmethod
    def _linsolve(self):
        """
        Solve linear system.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod
    def _nonlin_solve(self):
        """
        Solve nonlinear problem.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod
    def solve(self):
        """
        Solve generic problem. This calls _solve or _nonlinsolve depending on 
        the function arguments.
        
        Returns
        -------
        None
        
        """
        ...
    
    @abstractmethod    
    def time_evolve(self):
        """
        Make a single timestep.
        
        Returns
        -------
        None
        
        """
    
    @abstractmethod
    def sources(self):
        """
        Create physics problem specific sources (e. g. strain, gravity).
        
        Returns
        -------
        None
        
        """
        ...
    

def assemble_matrix(sK: np.ndarray, iK: np.ndarray, jK: np.ndarray,
                    ndof: int, solver: str, springs: Union[None,List]
                    ) -> csc_array:
    """
    Assemble matrix from indices.

    Parameters
    ----------
    sK : np.ndarray
        element degree of freedom matrix.
    sK : np.ndarray
        matrix values.
    iK : np.ndarray
        matrix row indices.
    jK : np.ndarray
        matrix column indices.
    ndof : int
        number of degrees of freedom
    solver : str
        solver used to solve the linear system.
    springs : list
        contains two np.ndarrays. The first one contains the indices of the
        degrees of freedom to which the springs are attached. The second one
        contains the spring constants.

    Returns
    -------
    M : scipy.sparse.csc_array, shape (ndof,ndof)
        assembled matrix.
    """
    #
    M = coo_array((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()
    # attach springs to dofs if there
    if springs:
        inds,spring_const = springs
        for i,k in zip(inds,spring_const):
            M[i,i] += k
    return M

def deleterowcol(A: csc_array, 
                 delrow: np.ndarray, 
                 delcol: np.ndarray) -> coo_array:
    """
    Copied from the topopt_cholmod.py code by Niels Aage and Villads Egede.

    Parameters
    ----------
    A : scipy.sparse.sparray shape (nodf,ndof)
        matrix.
    delrow : np.ndarray
        rows to delete .
    delcol : np.ndarray
        columns to delete .

    Returns
    -------
    A : coo_array shape (ndof_new,ndof_new)
        new matrix with rows and columns deleted.
    """
    # Assumes that matrix is in symmetric csc form
    m = A.shape[0]
    keep = np.delete(np.arange(0, m), delrow)
    A = A[keep, :]
    keep = np.delete(np.arange(0, m), delcol)
    A = A[:, keep]
    return A.tocoo()

def apply_bc(K: sparray, solver: str,
             free: Union[None,np.ndarray] = None,
             fixed: Union[None,np.ndarray] = None) -> Union[sparray,spmatrix]:
    """
    Apply boundary conditions to matrix K. At the moment only implements 
    Dirichlet boundary conditions equal to zero.

    Parameters
    ----------
    K : scipy.sparse.sparray shape (nodf,ndof)
        matrix.
    free : np.ndarray
        indices of variables for which we solve.
    fixed : np.ndarray
        indices of variables to be set to zero.

    Returns
    -------
    K : scipy.sparse.sparray or cvxopt.spmatrix shape (ndof_new,ndof_new)
        new matrix with applied boundary conditions.
    """
    if "scipy" in solver or "topoptlab" in solver:
        #
        K = K[free, :][:, free]
    if "cvxopt" in solver:
        # remove constrained dofs from matrix and convert to coo
        K = deleterowcol(K,fixed,fixed).tocoo()
        # solve system
        K = spmatrix(K.data,K.row,K.col)
    return K

def create_matrixinds(edofMat: np.ndarray,
                      mode: str = "full") -> Tuple[np.ndarray,np.ndarray]:
    """
    Create matrix indices to set up FE linear system / matrix.

    Parameters
    ----------
    edofMat : np.ndarray (nel,n_nodedof)
        element degree of freedom matrix.
    mode : str
        construct .

    Returns
    -------
    iM : np.ndarray shape (N)
        row indices for matrix construction.
    jM : np.ndarray shape (N)
        column indices for matrix construction.
    """
    #
    if mode == "full":
        iM = np.tile(edofMat,edofMat.shape[1])
        jM = np.repeat(edofMat,edofMat.shape[1])
    elif mode == "lower":
        #
        iM = [edofMat[:,i:] for i in np.arange(edofMat.shape[1])]
        iM = np.column_stack(iM)
        #
        jM = np.repeat(edofMat,np.arange(edofMat.shape[1],0,-1),axis=1)
        # sort
        mask = iM < jM
        iM[mask],jM[mask] = jM[mask],iM[mask]
    return iM.reshape(np.prod(iM.shape)),jM.reshape(np.prod(iM.shape))

def update_indices(indices: np.ndarray,
                   fixed: np.ndarray, 
                   mask: np.ndarray) -> np.ndarray:
    """
    Update the indices for the stiffness matrix construction by kicking out
    the fixed degrees of freedom and renumbering the indices. This is useful
    only if just one set of boundary conditions needs to be solved.

    Parameters
    ----------
    indices : np.ndarray
        indices of degrees of freedom used to construct the stiffness matrix.
    fixed : np.ndarray
        indices of fixed degrees of freedom.
    mask : np.ndarray
        mask to kick out fixed degrees of freedom.

    Returns
    -------
    indices : np.ndarray
        updated indices.

    """
    val, ind = np.unique(indices,return_inverse=True)

    _mask = ~np.isin(val, fixed)
    val[_mask] = np.arange(_mask.sum())

    return val[ind][mask]

def interpolate(ue: np.ndarray, xi: np.ndarray, eta: np.ndarray,
                zeta: Union[None,np.ndarray] = None,
                shape_functions: Callable = shape_functions) -> np.ndarray:
    """
    Interpolate node values in each element. Coordinates are assumed to be
    in the reference domain.

    Parameters
    ----------
    ue : np.ndarray,shape (nels,nedof).
        node values used for interpolation.
    xi : np.ndarray
        x coordinate of shape (nels). Coordinates are assumed to be
        in the reference domain.
    eta : np.ndarray
        y coordinate of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    zeta : np.ndarray or None
        z coordinate of shape (ncoords). Coordinates are assumed to be
        in the reference domain.
    shape_functions : callable
        shape functions of respective element
    Returns
    -------
    u : np.ndarray, shape (nels,nnodedof)
        interpolated state variable.

    """
    # interpolate
    interpolation = shape_functions(xi=xi,eta=eta,zeta=zeta)
    # get parameters for reshaping to desired end shape
    nshapef = interpolation.shape[1]
    nnodedof = int(ue.shape[1]/nshapef)
    u = ue * np.repeat(interpolation, nnodedof)[None,:]
    u = u.dot(np.tile(np.eye(nnodedof),(nshapef,1)))
    return u

def get_integrpoints(ndim: int, nq: int, 
                     method: str) -> Tuple[np.ndarray,np.ndarray]:
    """
    Get integration points and weights for numerical quadrature of integrals in
    interval [-1,1].

    Parameters
    ----------
    ndim : int
        number of spatial dimensions.
    nq : int
        number of integration/quadrature points.
    method : str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Currently only 'gauss-legendre',
        'gauss-hermite', 'gauss-chebyshev' and 'gauss-laguerre' supported as
        str.

    Returns
    -------
    x : np.ndarray, shape (nq,ndim)
        coordinates of quadrature point.
    w : np.ndarray, shape (nq)
        weight of quadrature points.
    """
    if method == "gauss-legendre":
        x,w = np.polynomial.legendre.leggauss(nq)
    elif method == "gauss-hermite":
        x,w = np.polynomial.hermite.hermgauss(nq)
    elif method == "gauss-chebyshev":
        x,w = np.polynomial.chebyshev.chebgauss(nq)
    elif method == "gauss-laguerre":
        x,w = np.polynomial.laguerre.laggauss(nq)
    elif hasattr(method, '__call__'):
        x,w = method(nq)
    else:
        raise ValueError("Invalid quadrature method.")
    # generate grid of points
    x = np.array(list(product(x,repeat=ndim)))
    w = np.prod(np.array(list(product(w,repeat=ndim))),axis=1)
    return x,w
