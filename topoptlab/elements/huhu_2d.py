# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any

import numpy as np

from topoptlab.elements.strain_measures import disp_gradient
from topoptlab.elements.bilinear_quadrilateral import shape_functions_hessian,invjacobian
from topoptlab.fem import get_integrpoints

def _lk_huhu_2d(xe: np.ndarray, 
                ue: np.ndarray,
                a: np.ndarray,
                kr: np.ndarray,     
                quadr_method: str = "gauss-legendre",
                t: np.ndarray = np.array([1.]),
                nquad: int = 2,
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
    a : np.ndarray, shape (nels,1) or
        exponent.
    kr : np.ndarray, shape (nels,1) or
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
    if len(a.shape) == 1:
        a = a[None,:]
    #
    if len(kr.shape) == 1:
        kr = kr[None,:]
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
    hessian = shape_functions_hessian(xi=xi, eta=eta) # (nq,n_basis,2,2)
    # apply isop. map
    hessian = Jinv.transpose((0,1,3,2))[:,:,None,:,:]@hessian[None,:,:,:]@\
              Jinv[:,:,None,:,:]#.transpose((0,1,3,2))
    # flatten hessian
    hessian = hessian.reshape((nel,nq,ndim**2))
    import sys 
    sys.exit()
    # calculate def. grad
    B,detJ = disp_gradient(xi=xi, eta=eta, xe=xe,
                           all_elems=True,
                           return_detJ=True)
    detJ = detJ.reshape(nel,nq)
    B = B.reshape(nel, nq,  B.shape[-2], B.shape[-1])
    # calculate displacement gradient and determinant
    #F = B@
    
    
    #
    integral = 1#B.transpose([0,1,3,2])@c[:,None,:,:]@B
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    # multiply thickness
    return t[:,None,None] * Ke

if __name__ == "__main__":
    _lk_huhu_2d(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                               [[-1,-1],[1,-1],[1,1],[-1,1]]]),
                a=1., kr=1.)