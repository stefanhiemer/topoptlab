# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.strain_measures import dispgrad_matrix
from topoptlab.elements.trilinear_hexahedron import shape_functions_dxi,\
                                                    shape_functions_hessian
from topoptlab.elements.isoparam_mapping import invjacobian

from topoptlab.fem import get_integrpoints

def _lk_huhu_3d(xe: np.ndarray, 
                ue: np.ndarray,
                exponent: np.ndarray,
                kr: np.ndarray,
                mode="newton",
                quadr_method: str = "gauss-lobatto",
                nquad: int = 3,
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 3D HuHu regularization with
    trilinear hexahedral Lagrangian elements:
        
        eng_dens = kr*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the two 
    parameters a 

    Parameters
    ----------
    xe : np.ndarray, shape (nels,8,3)
        coordinates of element nodes. Please look at the definition/function of 
        the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,24).
        nodal displacements.
    exponent : np.ndarray, shape (nels) or float
        exponent.
    kr : np.ndarray, shape (nels) or float
        regularization strength.
    quadr_method: str or callable
        name of quadrature method or function/callable that returns coordinates of
        quadrature points and weights. Check function get_integrpoints for
        available options.
    nquad : int
        number of quadrature points
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,24,24)
        element stiffness matrix.

    """
    #
    ndim=3
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    nel = xe.shape[0]
    #
    if isinstance(exponent,float):
        exponent = np.array([exponent])
    #
    if isinstance(kr,float):
        kr = np.array([kr]) 
    # get integration points
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, ndim,axis=1)]
    # get jacobian for isoparametric map
    Jinv,detJ = invjacobian(xi=xi,eta=eta,zeta=zeta,
                            xe=xe,
                            shape_functions_dxi=shape_functions_dxi,
                            all_elems=True,
                            return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    # collect hessian in ref. space
    B_hessian = shape_functions_hessian(xi=xi, 
                                        eta=eta, 
                                        zeta=zeta) # (nq,n_basis,3,3)
    # apply isop. map
    B_hessian = Jinv.transpose((0,1,3,2))[:,:,None,:,:]@B_hessian[None,:,:,:]@\
                Jinv[:,:,None,:,:]
    # flatten hessian
    B_hessian = B_hessian.reshape(B_hessian.shape[:3]+tuple([ndim**2]))
    B_hessian = B_hessian.transpose( (0,1,3,2) )
    # convert to hessian of a vector field
    B_hessian = np.kron(B_hessian,np.eye(ndim))
    # calculate def. grad
    B_F = dispgrad_matrix(xi=xi, eta=eta, zeta=zeta, xe=xe,
                          shape_functions_dxi=shape_functions_dxi,
                          invjacobian=invjacobian,
                          all_elems=True,
                          return_detJ=False) 
    B_F = B_F.reshape(nel, nq,  B_F.shape[-2], B_F.shape[-1])
    F = (B_F@ue[:,None,:,None]).reshape(nel,nq,ndim,ndim) + np.eye(ndim)[None,None,:,:]
    Fdet = np.linalg.det(F)
    if mode == "newton":
        finv = np.linalg.inv(F).transpose((0,1,3,2)).reshape((nel,nq,1,ndim**2))
        #
        integral = B_hessian.transpose([0,1,3,2])@B_hessian - \
                   ((exponent[:,None]*Fdet[:,:])[:,:,None,None]*\
                   B_hessian.transpose([0,1,3,2])@B_hessian@ue[:,None,:,None]@\
                   finv@B_F)
        integral = np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]\
                   *integral
    elif mode == "picard":
        integral = np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]\
                   *B_hessian.transpose([0,1,3,2])@B_hessian 
    # multiply by determinant and quadrature
    return (kr[:,None,None,None]*w[None,:,None,None]*integral*\
            detJ[:,:,None,None]).sum(axis=1)
    