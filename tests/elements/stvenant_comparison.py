# SPDX-License-Identifier: GPL-3.0-or-later
from typing import Any,Callable,Dict,Union

import numpy as np

from topoptlab.voigt import from_voigt
from topoptlab.elements.strain_measures import dispgrad_matrix,\
                                               lagrangian_strainvar_matrix
from topoptlab.elements.bilinear_quadrilateral import shape_functions_dxi
from topoptlab.elements.isoparam_mapping import invjacobian
from topoptlab.fem import get_integrpoints
from topoptlab.material_models.stvenant import stvenant_matmodel

def dummy_comparison(xe: np.ndarray,
                     ue: np.ndarray,
                     material_model: Callable,
                     material_constants: Dict,
                     quadr_method: str = "gauss-legendre",
                     t: np.ndarray = np.array([1.]),
                     nquad: int = 2,
                     shape_functions_grad: Callable = shape_functions_dxi,
                     invjac: Callable = invjacobian, 
                     ndim: int = 2,
                     **kwargs: Any) -> np.ndarray:
    """
    Coding pages 129/130 from the Wriggers book.
    
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
    shape_functions_grad : Callable
        gradient of shape functions of shape (n_nodes,ndim).
    
    Returns
    -------
    Kt : np.ndarray
        tangential stiffness matrix of shape (nel,n_nodes*ndim,n_nodes*ndim).
    fe : np.ndarray, 
        internal force vector of shape (nel,n_nodes*ndim).
    """
    #
    if len(xe.shape) == 2:
        xe = xe[None,:,:]
    #
    nel,n_nodes,ndim = xe.shape
    #
    if isinstance(t,float):
        t = np.array([t])
    #
    x,w=get_integrpoints(ndim=ndim,nq=nquad,method=quadr_method)
    nq =w.shape[0]
    #
    xi,eta,zeta = [_x[:,0] for _x in np.split(x, ndim,axis=1)] + \
                  (2*[None])[:(3-ndim)]
    #
    Jinv,detJ = invjac(xi=xi,eta=eta,zeta=zeta,xe=xe,
                       shape_functions_dxi=shape_functions_grad,
                       all_elems=True,return_det=True)
    Jinv = Jinv.reshape(nel,nq,ndim,ndim)
    detJ = detJ.reshape(nel,nq)
    #
    N_dx = shape_functions_dxi(xi=xi,eta=eta)[None,:,:,:]@Jinv.transpose((0,1,3,2))
    #
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
                                       all_elems=True,
                                       return_detJ=False)
    B_dE = B_dE.reshape(nel, nq,  B_dE.shape[-2], B_dE.shape[-1])
    # calculate constitutive tensor and 2.PK stress in Voigt notation
    s,c = material_model(F=F,**material_constants)
    # convert 2. PK stress from Voigt to tensor notation
    S = from_voigt(s,eng_conv=False)
    # constitutive part
    mat = B_dE.transpose([0,1,3,2])@c@B_dE
    geo = np.zeros(mat.shape)
    for i in range(n_nodes):
        for j in range(n_nodes):
            geo[:,:,i*ndim : (i+1)*ndim,j*ndim : (j+1)*ndim] += \
                np.kron(np.eye(2), N_dx[:,:,i,:,None].transpose([0,1,3,2])@S@N_dx[:,:,j,:,None])
    integral = mat+geo
    # multiply by determinant and quadrature
    Ke = (w[None,:,None,None]*integral*detJ[:,:,None,None]).sum(axis=1)
    #print("geo: ", (w[None,:,None,None]*geo*detJ[:,:,None,None]).sum(axis=1))
    #print("material: ", (w[None,:,None,None]*mat*detJ[:,:,None,None]).sum(axis=1))
    # multiply thickness
    if ndim == 2:
        Ke = t[:,None,None] * Ke
    return Ke 

if __name__ == "__main__":
    #
    from topoptlab.stiffness_tensors import isotropic_2d
    #
    nel = 2
    print(dummy_comparison(xe = np.array([[[-1,-1],[1,-1],[1,1],[-1,1]]]),
                           material_constants= {"c": np.ones((3,3)), 
                                                "h": np.ones((nel,1)),
                                                "mu": np.ones((nel,1))},
                           material_model=stvenant_matmodel,
                           ue = np.array([[0.,0.,0.,1.,
                                           0.,0.,0.,1.],
                                          ])))
