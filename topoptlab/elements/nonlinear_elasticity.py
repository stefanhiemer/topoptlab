# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any
from collections.abc import Callable

import numpy as np

from topoptlab.voigt import from_voigt
from topoptlab.fem import get_integrpoints 
from topoptlab.elements.isoparam_mapping import invjacobian as _invjacobian
from topoptlab.elements.strain_measures import dispgrad_matrix,\
                                               lagrangian_strainvar_matrix

def _lk_nonlinear_elast(xe: np.ndarray,
                        ue: np.ndarray,
                        material_model: Callable,
                        material_constants: dict,
                        quadr_method: str = "gauss-legendre",
                        t: np.ndarray = np.array([1.]),
                        nquad: int = 2,
                        shape_functions_dxi: None | Callable = None,
                        invjacobian: Callable = _invjacobian, 
                        **kwargs: Any) -> np.ndarray:
    """
    Create element tangential stiffness matrix for nonlinear elasticity with
    Lagrangian elements in terms of 2. Piola Kirchhoff stress S. Constitutive tensor must be derived by:
        
        C_ij = d S_i / d E_j
        
    E_j is the Lagrange strain.

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,n_nodes*ndim).
        nodal displacements.
    material_model: callable
        returns 2. PK stress tensor and constitutive tensor both in Voigt 
        notation as function of deformation gradient F (matrix form) and 
        the material constants provided in the dictionary. Outputs should 
        have shape (nel,nq,...).
    material_constants : dict
        contains the material constants needed to calculate 2. PK stress and 
        constitutive tensor. Keys must match arguments of material_model.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    t : np.ndarray of shape (nels) or (1)
        thickness of element in 2D or cross section area in 1D. Ignored
        in 3D.
    nquad : int
        number of quadrature points.
    shape_functions_dxi : callable or None
        gradients of shape functions of shape (...,n_nodes,ndim). If None,
        bilinear quadrilateral or trilinear hexahedral elements are assumed
        (ndim=2/3).
    invjacobian : callable
        inverse jacobian of parametric mapping of shape (...,ndim,ndim).
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,n_nodes*ndim,n_nodes*ndim)
        element stiffness matrix.
    fe : np.ndarray, shape (nels,n_nodes*ndim)
        element internal force vector.

    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel,n_nodes,ndim = xe.shape
    #
    # thickness (2D) or cross section area (1D)
    if ndim < 3:
        t = np.full(nel, t) if np.size(t) == 1 else np.asarray(t)
    # gradient of shape functions
    if shape_functions_dxi is None:
        if ndim == 1:
            from topoptlab.elements.linear_interval import shape_functions_dxi
        elif ndim == 2:
            from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
        elif ndim == 3:
            from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi
        else:
            raise NotImplementedError(f"only dimensions 1, 2 and 3 implemented. current ndim {ndim}")
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, ndim,axis=1)] + \
                  (2*[None])[:(3-ndim)]
    # calculate displacement gradient 
    B_h,detJ = dispgrad_matrix(xi=xi, eta=eta, zeta=zeta, xe=xe,
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
                                       zeta=zeta,
                                       xe=xe,
                                       F=F.reshape(nel*nq,ndim,ndim),
                                       shape_functions_dxi=shape_functions_dxi,
                                       invjacobian=invjacobian,
                                       all_elems=True,
                                       return_detJ=False)
    B_dE = B_dE.reshape(nel, nq,  B_dE.shape[-2], B_dE.shape[-1])
    # calculate constitutive tensor and 2.PK stress in Voigt notation
    s,c = material_model(F=F,**material_constants)
    # internal force
    fe = (B_dE.transpose(0,1,3,2) @ s[..., None]).squeeze(-1)
    # calculate constitutive tensor
    S = np.kron(np.eye(ndim),from_voigt(s, eng_conv=False))
    # constitutive part
    integral = B_dE.transpose([0,1,3,2])@c@B_dE
    # geometric part
    integral += B_h.transpose([0,1,3,2])@S@B_h # finish here
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    fe = (w[None,:,None]*fe*detJ[:,:,None]).sum(axis=1)
    # multiply thickness
    if ndim < 3:
        return t[:,None,None] * Ke, t[:,None] * fe
    else:
        return Ke, fe