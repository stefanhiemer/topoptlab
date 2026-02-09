# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Union

import numpy as np

from topoptlab.elements.strain_measures import dispgrad_matrix
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi,\
                                                      shape_functions_hessian,\
                                                      invjacobian
from topoptlab.fem import get_integrpoints

def _lk_huhu_2d(xe: np.ndarray, 
                ue: np.ndarray,
                exponent: Union[None,np.ndarray],
                kr: np.ndarray,
                mode="newton",
                quadr_method: str = "gauss-lobatto",
                t: np.ndarray = np.array([1.]),
                nquad: int = 3,
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D HuHu regularization with
    bilinear quadrilateral Lagrangian elements:
        
        eng_dens = kr/2*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the 
    regularizations strength kr and exponent a. If A is None, then instead:
        
        eng_dens = kr/2*(Hu)^T Hu

    Parameters
    ----------
    xe : np.ndarray, shape (nels,4,2)
        coordinates of element nodes. Please look at the
        definition/function of the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,8).
        nodal displacements.
    exponent : None or np.ndarray
        exponent with shape (nels), float or None. If None, the exponential 
        part is ignored.
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
    # get integration points
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta = [_x[:,0] for _x in np.split(x, ndim,axis=1)]
    # get jacobian for isoparametric map
    Jinv,detJ = invjacobian(xi=xi,eta=eta,xe=xe,
                            all_elems=True,return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    # collect hessian in ref. space
    B_hessian = shape_functions_hessian(xi=xi, eta=eta) # (nq,n_basis,2,2)
    # apply isop. map
    B_hessian = Jinv.transpose((0,1,3,2))[:,:,None,:,:]@B_hessian[None,:,:,:]@\
                Jinv[:,:,None,:,:]
    # flatten hessian
    B_hessian = B_hessian.reshape(B_hessian.shape[:3]+tuple([ndim**2]))
    B_hessian = B_hessian.transpose((0,1,3,2))
    # convert to hessian of a vector field
    B_hessian = np.kron(B_hessian,np.eye(ndim))
    #
    if exponent:
        # calculate def. grad
        B_F = dispgrad_matrix(xi=xi, eta=eta, zeta=None, xe=xe,
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
    else:
        integral = B_hessian.transpose([0,1,3,2])@B_hessian 
    # multiply by determinant and quadrature
    Ke = (kr[:,None,None,None]*w[None,:,None,None]*integral*detJ[:,:,None,None]\
          ).sum(axis=1)
    # multiply thickness
    return t[:,None,None] * Ke

def lk_huhu_2d(kr: np.ndarray = np.array([0,-1]),
               l: np.ndarray = np.array([1.,1.]),
               g: np.ndarray = np.array([0.]),
               t: float = 1.,
               **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 2D HuHu regularization with
    bilinear quadrilateral Lagrangian elements:
        
        eng_dens = kr*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the 
    regularizations strength kr and exponent a. If A is None, then instead:
        
        eng_dens = kr*(Hu)^T Hu

    Parameters
    ----------
    kr : float or np.ndarray 
        regularization strength.
    l : np.ndarray (2)
        side length of element.
    g : np.ndarray (1)
        angle of parallelogram.
    t : float
        thickness of element.
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,8,8)
        element stiffness matrix.

    """ 
    #
    tang = np.tan(g[0])
    # multiply thickness
    return t*kr*np.array([[0, 0, 0, 0],
                          [1, -1, 1, -1],
                          [1, -1, 1, -1],
                          [-2*tang, 2*tang, -2*tang, 2*tang]]) 

if __name__ == "__main__":
    
    xe=np.array([[[-1,-1],[1,-1],[1,1],[-1,1]],
                 [[-2.1,-2],[2.1,-2],[2,2.1],[-2,2]]])
    ue = np.array([[0.,0.,
                    0.,0.,
                    0.1,0.2,
                    0.,0.],
                   [0.,0.,
                    0.,0.,
                    0.1,0.2,
                    0.,0.]])
    #np.savetxt("huhu2d.csv", 
    #           _lk_huhu_2d(xe=xe,ue=ue,
    #                       exponent=None, 
    #                       kr=1.).flatten(), 
    #           delimiter=",")
    print(_lk_huhu_2d(xe=xe,ue=ue,
                           exponent=1., 
                           kr=1.))