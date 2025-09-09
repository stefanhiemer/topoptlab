# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Callable, Tuple, Union, Any

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import factorized 

from topoptlab.fem import create_matrixinds
from topoptlab.elements.screenedpoisson_2d import lk_screened_poisson_2d
from topoptlab.elements.screenedpoisson_3d import lk_screened_poisson_3d
from topoptlab.elements.bilinear_quadrilateral import create_edofMat as create_edofMat2d
from topoptlab.elements.trilinear_hexahedron import create_edofMat as create_edofMat3d
from topoptlab.filter.filter import TOFilter


class HelmholtzFilter(TOFilter):
    """
    Implements the Helmholtz filter from 
    
    Lazarov, Boyan Stefanov, and Ole Sigmund. "Filters in topology optimization 
    based on Helmholtz‐type differential equations." International journal for 
    numerical methods in engineering 86.6 (2011): 765-781.
    
    Implementation here is based on the implementation in 
    
    Andreassen, Erik, et al. "Efficient topology optimization in MATLAB using 
    88 lines of code." Structural and Multidisciplinary Optimization 43.1 
    (2011): 1-16.
    
    but extended to 3D.
    """
    
    def __init__(self,
                 nelx: int, nely: int, rmin: float,
                 nelz: Union[int, None] = None, 
                 l: np.ndarray = np.array([1.,1.]),
                 **kwargs: Any) -> None:
        """
        Assemble Helmholtz PDE based filter from "Efficient topology 
        optimization in MATLAB using 88 lines of code". Sets up the FEM problem
        and solves it via cholesky decomposition.
        
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
        l : np.ndarray
            side lengths of element
        
        Returns
        -------
        None

        """
        KF, self.TF = assemble_helmholtz_filter(nelx = nelx, 
                                                nely = nely, 
                                                rmin = rmin,
                                                nelz = nelz, 
                                                l = l)
        self.lu_solve = factorized(KF)
        
    def apply_filter(self, x: np.ndarray) -> np.ndarray:
        """
        Apply filter to the (intermediate) design variables x. Corresponds to
        distributing the element-based design variables onto the nodes of the 
        mesh via the transfer operator TF 
        
           xn = TF@x
        
        then solving the FE problem where the un-filtered nodal variables xn 
        enter as von-Neumann boundary condition/source terms
            
           K@xn_filtered = xn
        
        and recovering element-wise design variables via  
           
           x_filtered = TF.T@xn_filtered
        
        Parameters
        ----------
        x : np.ndarray
            (intermediate) design variables.

        Returns
        -------
        x_filtered : np.ndarray
            filtered design variables.

        """
        return self.TF.T @ self.lu_solve(self.TF@x) #x
        
    def apply_filter_dx(self, 
                        x_filtered : np.ndarray, 
                        dx_filtered : np.ndarray) -> np.ndarray:
        """
        Apply filter to the sensitivities with respect to filtered variables 
        x_filtered using the chain rule. The same trick is used as for applying 
        the filter to the design variables: it corresponds to distributing the 
        sensitivities onto the nodes of the mesh 
        
            dxn_filtered = TF@dx_filtered
        
        then solving the FE problem where the sensitivities with regards to 
        the filtered variables enter as von-Neumann boundary condition/source 
        terms
            
            K dxn = dxn_filtered
        
        and recovering element-wise sensitivities via  
           
           dx = TF.T@dxn
        
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
        return self.TF.T @ self.lu_solve(self.TF@dx_filtered) # TF.T @ self.lu_solve(TF@(dobj*xPhys))/np.maximum(0.001, x)

def assemble_helmholtz_filter(nelx: int, nely: int, rmin: float,
                              nelz: Union[int, None] = None, 
                              l: np.ndarray = np.array([1.,1.]),
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
    l : np.ndarray
        side lengths of element
        
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
