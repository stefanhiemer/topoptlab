# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any, Callable, Dict, Tuple

import numpy as np

from topoptlab.utils import from_voigt
from topoptlab.fem import get_integrpoints 
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
from topoptlab.elements.isoparam_mapping import invjacobian
from topoptlab.elements.strain_measures import dispgrad_matrix,\
                                               lagrangian_strainvar_matrix

def _lk_nonlinear_elast_2d(xe: np.ndarray,
                           ue: np.ndarray,
                           material_model: Callable,
                           material_constants: Dict,
                           quadr_method: str = "gauss-legendre",
                           t: np.ndarray = np.array([1.]),
                           nquad: int = 2,
                           **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D nonlinear elasticity with
    bilinear quadrilateral Lagrangian elements in terms of 2. Piola Kirchhoff
    stress S. Constitutive tensor must be derived by:
        
        C_ij = d S_i / d E_j
        
    E_j is the Lagrange strain.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,8).
        nodal displacements.
    material_model: callable
        returns 2. PK stress tensor and constitutive tensor both in Voigt 
        notation as function of deformation gradient F (matrix form) and 
        the material constants provided in the dictionary. Outputs should 
        have shape (nel,nq,...)
    material_constants : dict
        contains the material constants needed to calculate 2. PK stress and 
        constitutive tensor. Keys must match arguments of material_model.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    t : np.ndarray of shape (nels) or (1)
        thickness of element
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,8,8)
        element stiffness matrix.

    """
    ndim=2
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, ndim,axis=1)]
    # calculate displacement gradient 
    B_h,detJ = dispgrad_matrix(xi=xi, eta=eta, zeta=None, xe=xe,
                               shape_functions_dxi=shape_functions_dxi,
                               invjacobian=invjacobian,
                               all_elems=True,
                               return_detJ=True) 
    B_h = B_h.reshape(nel, nq,  B_h.shape[-2], B_h.shape[-1])
    detJ = detJ.reshape(nel,nq)
    # calculate convert to matrix form
    F = (B_h@ue[:,None,:,None]).reshape(nel,nq,ndim,ndim) \
         + np.eye(ndim)[None,None,:,:]
    #
    B_dE = lagrangian_strainvar_matrix(xi=xi, 
                                       eta=eta, 
                                       zeta=None,
                                       xe=xe,
                                       F=F.reshape(nel*nq,ndim,ndim),
                                       shape_functions_dxi=shape_functions_dxi,
                                       all_elems=True,
                                       return_detJ=False)
    B_dE = B_dE.reshape(nel, nq,  B_dE.shape[-2], B_dE.shape[-1])
    # calculate constitutive tensor and 2.PK stress in Voigt notation
    s,c = material_model(F=F,**material_constants)
    # calculate constitutive tensor
    S = np.kron(from_voigt(s),
                np.eye(ndim))
    # constitutive part
    integral = B_dE.transpose([0,1,3,2])@c@B_dE
    # geometric part
    integral += B_h.transpose([0,1,3,2])@S@B_h # finish here
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    # multiply thickness
    return t[:,None,None] * Ke

if __name__ == "__main__":
    
    from topoptlab.material_models.stvenant import stvenant_matmodel
    from topoptlab.material_models.neohooke import neohookean_matmodel
    
    #
    nel = 2
    print(_lk_nonlinear_elast_2d(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                               [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           material_constants= {"c": np.ones((nel,3,3)), 
                                                "h": np.ones((nel,1)),
                                                "mu": np.ones((nel,1))},
                           material_model=neohookean_matmodel,
                           ue = np.array([[0.,0.,0.,0.,
                                           0.,0.,0.,0.],
                                          [0.,0.,0.,0.,
                                           0.,0.,0.,0.],])))