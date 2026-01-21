# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.strain_measures import dispgrad_matrix
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi,\
                                                      shape_functions_hessian,\
                                                      invjacobian
from topoptlab.fem import get_integrpoints

def _lk_huhu_2d(xe: np.ndarray, 
                ue: np.ndarray,
                exponent: np.ndarray,
                kr: np.ndarray,
                mode="newton",
                quadr_method: str = "gauss-legendre",
                t: np.ndarray = np.array([1.]),
                nquad: int = 4,
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D HuHu regularization with
    bilinear quadrilateral Lagrangian elements:
        
        eng_dens = kr*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the two 
    parameters a 

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,8).
        nodal displacements.
    exponent : np.ndarray, shape (nels) or float
        exponent.
    kr : np.ndarray, shape (nels) or float
        regularization strength.
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
    #
    ndim=2
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
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=2,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, 2,axis=1)]
    # get jacobian for isoparametric map
    Jinv,detJ = invjacobian(xi=xi,eta=eta,xe=xe,
                            all_elems=True,return_det=True)
    Jinv = Jinv.reshape(nel,nq,2,2)
    detJ = detJ.reshape(nel,nq)
    # collect hessian in ref. space
    B_hessian = shape_functions_hessian(xi=xi, eta=eta) # (nq,n_basis,2,2)
    # apply isop. map
    B_hessian = Jinv.transpose((0,1,3,2))[:,:,None,:,:]@B_hessian[None,:,:,:]@\
                Jinv[:,:,None,:,:]#.transpose((0,1,3,2))
    #print(hessian.shape)
    # flatten hessian
    B_hessian = B_hessian.reshape(B_hessian.shape[:3]+tuple([ndim**2]))
    B_hessian = B_hessian.transpose( (0,1,3,2) )
    #print(hessian.shape)
    #print(hessian[0,0])
    B_hessian = np.kron(B_hessian,np.eye(ndim))
    #print(hessian[0,0])
    #print("B_hessian",B_hessian.shape)
    # calculate def. grad
    B_F = dispgrad_matrix(xi=xi, eta=eta, zeta=None, xe=xe,
                          shape_functions_dxi=shape_functions_dxi,
                          invjacobian=invjacobian,
                          all_elems=True,
                          return_detJ=False) 
    B_F = B_F.reshape(nel, nq,  B_F.shape[-2], B_F.shape[-1])
    #print("B_F ",B_F.shape)
    #print("ue ",ue.shape)
    F = (B_F@ue[:,None,:,None]).reshape(nel,nq,ndim,ndim) + np.eye(ndim)[None,None,:,:]
    #print("F ",F.shape)
    Fdet = np.linalg.det(F)
    #print("Fdet ",Fdet.shape)
    if mode == "newton":
        finv = np.linalg.inv(F).transpose((0,1,3,2)).reshape((nel,nq,1,ndim**2))
        #print("finv ",finv.shape)
        #
        
        #
        integral = B_hessian.transpose([0,1,3,2])@B_hessian - \
                   (exponent[:,None]*Fdet[:,:])[:,:,None,None]*\
                   B_hessian.transpose([0,1,3,2])@B_hessian@ue[:,None,:,None]@\
                   finv[:,:,None,:]@B_F
        integral = np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]\
                   *integral
    elif mode == "picard":
        integral = np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]\
                   *B_hessian.transpose([0,1,3,2])@B_hessian 
    #
    #print("integral ",integral.shape)
    # multiply by determinant and quadrature
    Ke = (kr[:,None,None,None]*w[None,:,None,None]*integral*detJ[:,:,None,None]\
          ).sum(axis=1)
    # multiply thickness
    #print("Ke ",Ke.shape)
    return t[:,None,None] * Ke

if __name__ == "__main__":
    _lk_huhu_2d(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                               [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                ue = np.array([[0.,0.,0.,0.,
                                0.,0.,0.,0.],
                               [0.,0.,0.,0.,
                                0.,0.,0.,0.],]),
                exponent=1., kr=1.)