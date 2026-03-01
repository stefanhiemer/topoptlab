# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,  Callable

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
                shape_functions_grad: Callable = shape_functions_dxi,
                invjac: Callable = invjacobian, 
                **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 3D HuHu regularization with
    trilinear hexahedral Lagrangian elements:
        
        eng_dens = kr*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the two 
    parameters a 

    Parameters
    ----------
    xe : np.ndarray, shape (nels,n_nodes,ndim)
        coordinates of element nodes. Please look at the definition/function of 
        the shape function, then the node ordering is
        clear.
    ue : np.ndarray,shape (nels,n_nodes*ndim).
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
        number of quadrature points.
    shape_functions_grad : Callable
        gradients of shape functions of shape (...,n_nodes,ndim).
    invjac : Callable
        inverse jacobian of parametric mapping of shape (...,ndim,ndim).
        
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
    Jinv,detJ = invjac(xi=xi,
                       eta=eta,
                       zeta=zeta,
                       xe=xe,
                       shape_functions_dxi=shape_functions_grad,
                       all_elems=True,
                       return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    # collect hessian in ref. space
    B_hessian = shape_functions_hessian(xi=xi, eta=eta, zeta=zeta) # (nq,n_basis,2,2)
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
        B_F = dispgrad_matrix(xi=xi, eta=eta, zeta=zeta, xe=xe,
                              shape_functions_dxi=shape_functions_grad,
                              invjacobian=invjacobian,
                              all_elems=True,
                              return_detJ=False) 
        B_F = B_F.reshape(nel, nq,  B_F.shape[-2], B_F.shape[-1])
        F = (B_F@ue[:,None,:,None]).reshape(nel,nq,ndim,ndim) + np.eye(ndim)[None,None,:,:]
        Fdet = np.linalg.det(F)
        # picard part of tangent stiffness matrix
        if mode in ["picard","newton"]:
            Ke = np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]\
                       *B_hessian.transpose([0,1,3,2])@B_hessian
        # inner forces
        fe = (Ke@ue[:,None,:,None])[...,0]
        # other part of tangent stiffness matrix needed for newton
        if mode == "newton":
            finv = np.linalg.inv(F).transpose((0,1,3,2)).reshape((nel,nq,1,ndim**2))
            #
            Ke = Ke - (np.exp(-exponent[:,None]*Fdet[:,:])[:,:,None,None]*\
                  (exponent[:,None]*Fdet[:,:])[:,:,None,None]*\
                   B_hessian.transpose([0,1,3,2])@B_hessian@ue[:,None,:,None]@\
                   finv@B_F)
    else:
        Ke = B_hessian.transpose([0,1,3,2])@B_hessian 
    # multiply by determinant and quadrature
    Ke = (kr[:,None,None,None]*w[None,:,None,None]*Ke*detJ[:,:,None,None]\
          ).sum(axis=1)
    # 
    if exponent:
        fe = (kr[:,None,None]*w[None,:,None]*fe*detJ[:,:,None]\
              ).sum(axis=1)
        return Ke, fe 
    else:
        return Ke

def lk_huhu_3d(kr: np.ndarray = np.array(1.),
               l: np.ndarray = np.array([1.,1.,1.]),
               g: np.ndarray = np.array([0.,0.]),
               **kwargs: Any) -> np.ndarray:
    """
    Create element stiffness matrix for 3D HuHu regularization with
    trilinear hexahedral Lagrangian elements:
        
        eng_dens = kr*exp(-a * det(F)) (Hu)^T Hu
    
    where H is the spatial hessian, F the deformation gradient with the 
    regularizations strength kr and exponent a. If A is None, then instead:
        
        eng_dens = kr*(Hu)^T Hu

    Parameters
    ----------
    kr : float or np.ndarray 
        regularization strength.
    l : np.ndarray (3)
        side length of element.
    g : np.ndarray (2)
        angles of parallelepiped.
        
    Returns
    -------
    Ke : np.ndarray, shape (nels,24,24)
        element stiffness matrix.

    """ 
    #
    return np.column_stack(((2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[2]*np.tan(g[1]) - 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-l[0]**2*l[1]*np.tan(g[0]) - 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2 - l[0]**2*l[2]**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[1]*np.tan(g[0]) + 2*l[1]*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**2*l[1]**2 + l[0]**2*l[2]**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (l[0]**2*l[2]*np.tan(g[1]) + 2*l[1]**2*l[2]*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**2*l[1]**2 - 2*l[0]**2*l[2]**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**2*l[1]**2 + 2*l[0]**2*l[2]**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                 (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                 0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                 0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[2]**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[2]**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (3*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 - 6*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-6*l[0]**4 - 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2 - 6*l[0]**2*l[2]**2*np.tan(g[1])**2 - 6*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-l[0]**4 - l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]))/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - l[0]**2*l[1]**2 + 3*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (l[0]**4 + l[0]**2*l[1]**2*np.tan(g[0])**2 + l[0]**2*l[1]**2*np.tan(g[0]) + l[0]**2*l[2]**2*np.tan(g[1])**2 + l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (-3*l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[1]**2 - 3*l[0]**2*l[2]**2*np.tan(g[1]) - 2*l[0]**2*l[2]**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 - 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]),
                  0,
                  0,
                  (2*l[0]**4 + 2*l[0]**2*l[1]**2*np.tan(g[0])**2 + 2*l[0]**2*l[2]**2*np.tan(g[1])**2 + 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]) + (-2*l[0]**4 - 2*l[0]**2*l[1]**2*np.tan(g[0])**2 - l[0]**2*l[1]**2*np.tan(g[0]) - 2*l[0]**2*l[2]**2*np.tan(g[1])**2 - l[0]**2*l[2]**2*np.tan(g[1]) - 4*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 - 2*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) - 2*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2)/(l[0]**3*l[1]*l[2]) + (6*l[0]**4 + 6*l[0]**2*l[1]**2*np.tan(g[0])**2 + 6*l[0]**2*l[1]**2*np.tan(g[0]) + 2*l[0]**2*l[1]**2 + 6*l[0]**2*l[2]**2*np.tan(g[1])**2 + 6*l[0]**2*l[2]**2*np.tan(g[1]) + 2*l[0]**2*l[2]**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])**2*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[0])**2 + 12*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1])**2 + 6*l[1]**2*l[2]**2*np.tan(g[0])*np.tan(g[1]) + 4*l[1]**2*l[2]**2*np.tan(g[1])**2)/(3*l[0]**3*l[1]*l[2]))).reshape(-1,24,24)                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                 
if __name__ == "__main__":
    
    #
    xe = np.array([[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1], 
                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],
                   [[-2.1,-2,-2],[2.1,-2,-2],
                    [2,2.1,-2],[-2,2,-2], 
                    [-2.,-2,2],[2.,-2,2],
                    [2,2.,2],[-2,2,2]]])
    #
    ue = np.array([[0.,0.,0.,
                    0.,0.,0.,
                    0.1,0.2,0.15,
                    0.,0.,0., 
                    0.,0.,0.,
                    0.,0.,0.,
                    0.,0.,0.,
                    0.,0.,0.],
                   [0.,0.,0.,
                    0.,0.,0.,
                    0.1,0.2,0.15,
                    0.,0.,0.,
                    0.,0.,0.,
                    0.,0.,0.,
                    0.,0.,0.,
                    0.,0.,0.]])
    
    #
    Ke = _lk_huhu_3d(xe=xe,
                        ue=ue,
                        exponent=None, 
                        kr=1.)[0]
    
    print(Ke-lk_huhu_3d(l=np.array([2.,2.,2.])))
    #print(np.isclose(lk_huhu_3d(), 
    #                 _lk_huhu_3d(xe=xe,ue=ue,
    #                       exponent=0., 
    #                       kr=1.)[0:1]))